//! Decompression bombs must fail without allocating past the tensor.
//!
//! Each test writes a small tensor with the Zstd plane codec, then replaces
//! one Zstd payload in place with a frame of the same length that decodes
//! to 512 MiB of zeros (RLE blocks, padded with a Zstd skippable frame).
//! Every length and offset in the file stays valid, so the reader reaches
//! the codec. The global allocator records the largest single allocation;
//! the tests assert it stays far below the bomb's size. Before the codecs
//! took a bound, the reader decoded the whole frame and only then compared
//! its length with the chain.
//!
//! This binary holds only these tests: they share the allocator's counter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ptwm_core::chain::{Chain, ChainEdge, ChainNode, TerminalRef};
use ptwm_core::codec::{CodecId, codec_for};
use ptwm_core::compact::{compress_compact, decode_compact};
use ptwm_core::compressor::{CompressorOptions, InputTensor, compress_model};
use ptwm_core::container::ContainerReader;
use ptwm_core::layout::PlaneLayout;
use ptwm_core::transforms::op::OpId;
use ptwm_core::types::{Role, ValueFormat};

struct PeakAlloc;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

const BOMB_LEN: u64 = 512 << 20;
/// Largest single allocation a bomb test may cause.
const ALLOC_LIMIT: usize = 64 << 20;
const RAW_LEN: usize = 128 * 1024;

/// Random bytes in `0..16`: Zstd codes them at about half their size.
fn raw() -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..RAW_LEN)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x & 0x0F) as u8
        })
        .collect()
}

/// Source(int8) → BytePassthrough → one Value terminal.
fn passthrough_chain(n_bytes: u32) -> Chain {
    let role = Role::Value {
        format: ValueFormat::IntN { bits: 8 },
    };
    let mut source = vec![1u8];
    source.extend_from_slice(&n_bytes.to_le_bytes());
    source.extend_from_slice(&17u16.to_le_bytes());
    Chain {
        nodes: vec![
            ChainNode {
                op: OpId::Source,
                params: source,
            },
            ChainNode {
                op: OpId::BytePassthrough,
                params: Vec::new(),
            },
        ],
        edges: vec![ChainEdge {
            src_node: 0,
            src_output_idx: 0,
            dst_node: 1,
            dst_input_idx: 0,
            role_override: Some(role.clone()),
            vendor_bytes: Vec::new(),
        }],
        terminals: vec![TerminalRef {
            node_idx: 1,
            output_idx: 0,
            role,
        }],
    }
}

fn input(raw: &[u8]) -> InputTensor<'_> {
    InputTensor {
        name: "t".into(),
        candidate_chains: vec![passthrough_chain(raw.len() as u32)],
        dtype_code: 0,
        input_format: 1,
        raw_bytes: raw,
        shape: None,
        dtype_name: None,
        delta_reference_blake3: None,
    }
}

fn opts(chunk_size: Option<u32>) -> CompressorOptions {
    CompressorOptions {
        method_hint: 1,
        emit_payload_hash: false,
        emit_plane_crc: false,
        forced_codec: Some(CodecId::Zstd),
        allow_codec_ids: None,
        chunk_size,
    }
}

/// A Zstd frame of `BOMB_LEN` zeros from 128 KiB RLE blocks, followed by a
/// skippable frame that pads it to exactly `len` bytes.
fn bomb_of_len(len: usize) -> Vec<u8> {
    const BLOCK: u64 = 128 * 1024;
    let mut out = vec![0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x38];
    let mut left = BOMB_LEN;
    while left > 0 {
        let n = left.min(BLOCK);
        left -= n;
        let header = ((n as u32) << 3) | (1 << 1) | u32::from(left == 0);
        out.extend_from_slice(&header.to_le_bytes()[..3]);
        out.push(0);
    }
    let pad = len
        .checked_sub(out.len() + 8)
        .expect("payload too short for the bomb");
    out.extend_from_slice(&0x184D_2A50u32.to_le_bytes());
    out.extend_from_slice(&(pad as u32).to_le_bytes());
    out.resize(len, 0);
    out
}

/// Replace the one occurrence of `payload` in `file` with a bomb.
fn plant_bomb(file: &mut [u8], payload: &[u8]) {
    let at = file
        .windows(payload.len())
        .position(|w| w == payload)
        .expect("Zstd payload in file");
    assert!(
        file[at + 1..].windows(payload.len()).all(|w| w != payload),
        "payload must be unique"
    );
    file[at..at + payload.len()].copy_from_slice(&bomb_of_len(payload.len()));
}

fn zstd_payload(plane: &[u8]) -> Vec<u8> {
    codec_for(CodecId::Zstd)
        .unwrap()
        .encode(plane, None, &PlaneLayout::Flat)
        .unwrap()
        .payload
}

fn assert_bounded(err: String, t0: Instant) {
    assert!(err.contains("more than"), "{err}");
    assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(
        largest < ALLOC_LIMIT,
        "largest allocation was {largest} bytes"
    );
}

#[test]
fn compact_frame_bomb_is_bounded() {
    let raw = raw();
    let mut blob = compress_compact(&input(&raw), &opts(None))
        .unwrap()
        .expect("compact frame");
    assert_eq!(decode_compact(&blob).unwrap().raw, raw);
    plant_bomb(&mut blob, &zstd_payload(&raw));
    let t0 = Instant::now();
    let err = decode_compact(&blob).unwrap_err().to_string();
    assert_bounded(err, t0);
}

#[test]
fn legacy_plane_bomb_is_bounded() {
    let raw = raw();
    let mut buf = Vec::new();
    compress_model(Cursor::new(&mut buf), &[input(&raw)], opts(None)).unwrap();
    assert_eq!(
        ContainerReader::open(&buf)
            .unwrap()
            .decode_tensor("t")
            .unwrap(),
        raw
    );
    plant_bomb(&mut buf, &zstd_payload(&raw));
    let t0 = Instant::now();
    let err = ContainerReader::open(&buf)
        .unwrap()
        .decode_tensor("t")
        .unwrap_err()
        .to_string();
    assert_bounded(err, t0);
}

#[test]
fn legacy_chunk_bomb_is_bounded() {
    const CHUNK: usize = 64 * 1024;
    let raw = raw();
    let mut buf = Vec::new();
    compress_model(
        Cursor::new(&mut buf),
        &[input(&raw)],
        opts(Some(CHUNK as u32)),
    )
    .unwrap();
    assert_eq!(
        ContainerReader::open(&buf)
            .unwrap()
            .decode_tensor("t")
            .unwrap(),
        raw
    );
    plant_bomb(&mut buf, &zstd_payload(&raw[..CHUNK]));
    let t0 = Instant::now();
    let err = ContainerReader::open(&buf)
        .unwrap()
        .decode_tensor("t")
        .unwrap_err()
        .to_string();
    assert_bounded(err, t0);
}
