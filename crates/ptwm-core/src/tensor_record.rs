//! Tensor record serialization.
//!
//! `chain_ref: u16` selects the preprocessing chain for this tensor:
//! - High bit (`0x8000`) set: an inline chain payload follows the fixed
//!   header (before dependencies and per-terminal records).
//! - High bit clear: `chain_ref & 0x7FFF` is a chain registry index.

use crate::chain::Chain;
use crate::chain::wire::{read_chain, write_chain};
use crate::error::PtwmCoreError;
use crate::extension::ExtensionTableBuilder;
use crate::extension::table::ExtensionTable;
use crate::plane_record::{PlaneRecord, parse_plane_record, write_plane_record};

pub const TENSOR_FLAG_DEPENDENCIES: u8 = 0x01;
pub const TENSOR_FLAG_METADATA: u8 = 0x02;
pub const TENSOR_FLAG_PAYLOAD_HASH: u8 = 0x04;
/// High bit of `chain_ref`: when set, an inline chain follows the fixed header.
pub const CHAIN_REF_INLINE_BIT: u16 = 0x8000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RefKind {
    LocalTensor = 0x01,
    ExternalPtwm = 0x02,
    ExternalSafetensors = 0x03,
}

impl RefKind {
    pub fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0x01 => Some(Self::LocalTensor),
            0x02 => Some(Self::ExternalPtwm),
            0x03 => Some(Self::ExternalSafetensors),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    pub ref_kind: RefKind,
    pub expected_hash: Option<[u8; 32]>,
    pub ref_bytes: Vec<u8>,
}

/// A serialized tensor record.
///
/// `chain_ref` encodes the chain association:
/// - `chain_ref & 0x8000 == 0`: the tensor's chain is in the container chain
///   registry at index `chain_ref & 0x7FFF`. `inline_chain` must be `None`.
/// - `chain_ref & 0x8000 != 0`: the chain is stored inline in this record
///   (in `inline_chain`). The low 15 bits are a hint or zero.
///
/// `terminals` (formerly `planes`) hold the per-output codec records in
/// terminal order; `num_terminals` is derived from `terminals.len()` on write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRecord {
    /// Chain reference. See struct doc for bit layout.
    pub chain_ref: u16,
    pub dtype_code: u16,
    pub input_format: u8,
    pub flags: u8,
    pub orig_size: u64,
    pub name: String,
    pub payload_hash: Option<u64>,
    /// Inline chain payload, present iff `chain_ref & CHAIN_REF_INLINE_BIT != 0`.
    pub inline_chain: Option<Chain>,
    pub dependencies: Vec<Dependency>,
    pub tensor_metadata: Option<Vec<u8>>,
    /// Per-terminal (formerly per-plane) codec records.
    pub terminals: Vec<PlaneRecord>,
}

/// Compute how many padding bytes are needed after `name_bytes` (of length
/// `name_len`) so that the total length of `name_bytes` is a multiple of 4.
fn name_padding(name_len: usize) -> usize {
    let rem = name_len % 4;
    if rem == 0 { 0 } else { 4 - rem }
}

pub fn write_tensor_record(rec: &TensorRecord, out: &mut Vec<u8>) -> Result<(), PtwmCoreError> {
    // Validate inline_chain vs chain_ref consistency.
    let has_inline = rec.chain_ref & CHAIN_REF_INLINE_BIT != 0;
    if has_inline && rec.inline_chain.is_none() {
        return Err(PtwmCoreError::InvalidContainer(
            "chain_ref has INLINE bit set but inline_chain is None".into(),
        ));
    }
    if !has_inline && rec.inline_chain.is_some() {
        return Err(PtwmCoreError::InvalidContainer(
            "chain_ref has INLINE bit clear but inline_chain is Some".into(),
        ));
    }

    // Derive flags from struct fields (overrides caller-set flags).
    let mut flags: u8 = 0;
    if !rec.dependencies.is_empty() {
        flags |= TENSOR_FLAG_DEPENDENCIES;
    }
    if rec.tensor_metadata.is_some() {
        flags |= TENSOR_FLAG_METADATA;
    }
    if rec.payload_hash.is_some() {
        flags |= TENSOR_FLAG_PAYLOAD_HASH;
    }

    let num_terminals = rec.terminals.len() as u8;
    let name_bytes = rec.name.as_bytes();
    let name_len = name_bytes.len() as u16;

    // Fixed header: chain_ref(u16) + dtype_code(u16) + input_format(u8) +
    //               num_terminals(u8) + flags(u8) + reserved(u8) +
    //               orig_size(u64) + name_len(u16)  =  2+2+1+1+1+1+8+2 = 18 bytes
    out.extend_from_slice(&rec.chain_ref.to_le_bytes());
    out.extend_from_slice(&rec.dtype_code.to_le_bytes());
    out.push(rec.input_format);
    out.push(num_terminals);
    out.push(flags);
    out.push(0u8); // reserved
    out.extend_from_slice(&rec.orig_size.to_le_bytes());
    out.extend_from_slice(&name_len.to_le_bytes());

    // name_bytes + padding to multiple of 4
    out.extend_from_slice(name_bytes);
    let pad = name_padding(name_bytes.len());
    for _ in 0..pad {
        out.push(0u8);
    }

    // payload_hash  (present iff TENSOR_FLAG_PAYLOAD_HASH)
    if let Some(hash) = rec.payload_hash {
        out.extend_from_slice(&hash.to_le_bytes());
    }

    // inline chain  (present iff chain_ref & CHAIN_REF_INLINE_BIT)
    //
    // Wire layout: the inline chain section is self-contained.
    // A local extension table immediately precedes the chain bytes so the
    // reader can resolve u16 table indices without access to the file-level
    // Extension Table.
    //
    //   inline_et_count:  u32
    //   inline_et_entries: [ExtensionTableEntry; count]
    //   chain_bytes:      (num_nodes, ..., nodes_section, edges_section, terminals_section)
    if let Some(chain) = &rec.inline_chain {
        let mut builder = ExtensionTableBuilder::new();
        let mut chain_bytes = Vec::new();
        write_chain(chain, &mut builder, &mut chain_bytes)?;
        let inline_table = ExtensionTable {
            entries: builder.finish(),
        };
        let et_bytes = inline_table
            .to_bytes()
            .map_err(|e| PtwmCoreError::InvalidContainer(format!("inline chain et: {e}")))?;
        out.extend_from_slice(&et_bytes);
        out.extend_from_slice(&chain_bytes);
    }

    // dependencies  (present iff TENSOR_FLAG_DEPENDENCIES)
    if !rec.dependencies.is_empty() {
        let dep_count = rec.dependencies.len() as u16;
        out.extend_from_slice(&dep_count.to_le_bytes());
        for dep in &rec.dependencies {
            let ref_flags: u8 = if dep.expected_hash.is_some() {
                0x01
            } else {
                0x00
            };
            out.push(dep.ref_kind as u8);
            out.push(ref_flags);
            out.extend_from_slice(&(dep.ref_bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(&dep.ref_bytes);
            if let Some(hash) = dep.expected_hash {
                out.extend_from_slice(&hash);
            }
        }
    }

    // tensor_metadata  (present iff TENSOR_FLAG_METADATA)
    if let Some(meta) = &rec.tensor_metadata {
        out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        out.extend_from_slice(meta);
    }

    // per-terminal records (formerly plane records)
    for terminal in &rec.terminals {
        write_plane_record(terminal, out)?;
    }

    Ok(())
}

pub fn parse_tensor_record(buf: &[u8]) -> Result<(TensorRecord, usize), PtwmCoreError> {
    if buf.len() < 18 {
        return Err(PtwmCoreError::InvalidContainer(
            "tensor record header truncated".into(),
        ));
    }

    fn need(buf: &[u8], pos: usize, n: usize, what: &str) -> Result<(), PtwmCoreError> {
        if pos.checked_add(n).is_none_or(|end| end > buf.len()) {
            return Err(PtwmCoreError::InvalidContainer(format!(
                "tensor record truncated reading {what}"
            )));
        }
        Ok(())
    }

    let mut pos = 0usize;

    let chain_ref = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
    pos += 2;

    let dtype_code = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap());
    pos += 2;

    let input_format = buf[pos];
    pos += 1;

    let num_terminals = buf[pos] as usize;
    pos += 1;

    let flags = buf[pos];
    pos += 1;

    // reserved byte
    pos += 1;

    let orig_size = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
    pos += 8;

    let name_len = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;

    // name_bytes + padding
    need(buf, pos, name_len, "name")?;
    let name = std::str::from_utf8(&buf[pos..pos + name_len])
        .map_err(|_| PtwmCoreError::InvalidContainer("tensor name is not valid UTF-8".into()))?
        .to_string();
    pos += name_len;
    let pad = name_padding(name_len);
    need(buf, pos, pad, "name padding")?;
    pos += pad;

    // payload_hash
    let payload_hash = if flags & TENSOR_FLAG_PAYLOAD_HASH != 0 {
        need(buf, pos, 8, "payload_hash")?;
        let h = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        Some(h)
    } else {
        None
    };

    // inline chain (present iff chain_ref & CHAIN_REF_INLINE_BIT)
    //
    // Wire layout: a local extension table precedes the chain
    // bytes. Read it first, then pass it to `read_chain`.
    let inline_chain = if chain_ref & CHAIN_REF_INLINE_BIT != 0 {
        // Read the inline extension table (count: u32 + entries) directly
        // from the buffer slice via a Cursor. The cursor's position tells
        // us how many bytes the table consumed, so we don't need to
        // re-serialize it (which would also be an OOM surface for large
        // tables — count is already bounded by MAX_TABLE_ENTRIES, but
        // the re-serialize was redundant work either way).
        let mut cursor = std::io::Cursor::new(&buf[pos..]);
        let inline_table = ExtensionTable::read(&mut cursor).map_err(|e| {
            PtwmCoreError::InvalidContainer(format!("inline chain extension table: {e}"))
        })?;
        pos += cursor.position() as usize;
        let (chain, consumed) = read_chain(&buf[pos..], &inline_table).map_err(|e| {
            PtwmCoreError::InvalidContainer(format!("inline chain parse error: {e}"))
        })?;
        pos += consumed;
        Some(chain)
    } else {
        None
    };

    // dependencies
    let dependencies = if flags & TENSOR_FLAG_DEPENDENCIES != 0 {
        need(buf, pos, 2, "dep_count")?;
        let dep_count = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let mut deps = Vec::with_capacity(dep_count);
        for _ in 0..dep_count {
            need(buf, pos, 4, "dependency header")?;
            let ref_kind_raw = buf[pos];
            let ref_kind = RefKind::from_u8(ref_kind_raw).ok_or_else(|| {
                PtwmCoreError::InvalidContainer(format!("unknown ref_kind 0x{:02x}", ref_kind_raw))
            })?;
            pos += 1;
            let ref_flags = buf[pos];
            pos += 1;
            let ref_len = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            need(buf, pos, ref_len, "ref_bytes")?;
            let ref_bytes = buf[pos..pos + ref_len].to_vec();
            pos += ref_len;
            let expected_hash = if ref_flags & 0x01 != 0 {
                need(buf, pos, 32, "dependency hash")?;
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&buf[pos..pos + 32]);
                pos += 32;
                Some(hash)
            } else {
                None
            };
            deps.push(Dependency {
                ref_kind,
                expected_hash,
                ref_bytes,
            });
        }
        deps
    } else {
        Vec::new()
    };

    // tensor_metadata
    let tensor_metadata = if flags & TENSOR_FLAG_METADATA != 0 {
        need(buf, pos, 4, "metadata length")?;
        let kv_len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        need(buf, pos, kv_len, "metadata bytes")?;
        let kv_bytes = buf[pos..pos + kv_len].to_vec();
        pos += kv_len;
        Some(kv_bytes)
    } else {
        None
    };

    // per-terminal records (formerly plane records)
    let mut terminals = Vec::with_capacity(num_terminals);
    for _ in 0..num_terminals {
        let (plane, consumed) = parse_plane_record(&buf[pos..])?;
        pos += consumed;
        terminals.push(plane);
    }

    Ok((
        TensorRecord {
            chain_ref,
            dtype_code,
            input_format,
            flags,
            orig_size,
            name,
            payload_hash,
            inline_chain,
            dependencies,
            tensor_metadata,
            terminals,
        },
        pos,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{Chain, ChainNode, TerminalRef};
    use crate::codec::{CodecId, StateSource};
    use crate::layout::PlaneLayout;
    use crate::transforms::op::OpId;
    use crate::types::PlaneRole;
    use crate::types::role::{Role, ValueFormat};

    fn identity_plane() -> PlaneRecord {
        PlaneRecord {
            role: PlaneRole::Value,
            codec_id: CodecId::Identity,
            codec_table_idx: 0,
            state_source: StateSource::None,
            state_version: 0,
            state_info: 0,
            payload_len: 0,
            crc32: None,
            chunk_table: None,
            external_state: None,
            inline_state_bytes: Vec::new(),
            payload_bytes: Vec::new(),
            layout: PlaneLayout::Flat,
        }
    }

    fn minimal_chain() -> Chain {
        Chain {
            nodes: vec![ChainNode {
                op: OpId::BytePassthrough,
                params: vec![],
            }],
            edges: vec![],
            terminals: vec![TerminalRef {
                node_idx: 0,
                output_idx: 0,
                role: Role::Value {
                    format: ValueFormat::Fp4E2m1,
                },
            }],
        }
    }

    #[test]
    fn minimal_record_roundtrip() {
        let rec = TensorRecord {
            chain_ref: 0x0001, // registry ref, no inline bit
            dtype_code: 0x0001,
            input_format: 1,
            flags: 0,
            orig_size: 1024,
            name: "weight".to_string(),
            payload_hash: None,
            inline_chain: None,
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![identity_plane()],
        };
        let mut out = Vec::new();
        write_tensor_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_tensor_record(&out).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(parsed.chain_ref, rec.chain_ref);
        assert_eq!(parsed.dtype_code, rec.dtype_code);
        assert_eq!(parsed.input_format, rec.input_format);
        assert_eq!(parsed.orig_size, rec.orig_size);
        assert_eq!(parsed.name, rec.name);
        assert_eq!(parsed.payload_hash, rec.payload_hash);
        assert_eq!(parsed.inline_chain, rec.inline_chain);
        assert_eq!(parsed.dependencies, rec.dependencies);
        assert_eq!(parsed.tensor_metadata, rec.tensor_metadata);
        assert_eq!(parsed.terminals, rec.terminals);
    }

    #[test]
    fn tensor_record_with_registry_chain_roundtrip() {
        // chain_ref = 0x0003 (no high bit), inline_chain None.
        let rec = TensorRecord {
            chain_ref: 0x0003,
            dtype_code: 0x0002,
            input_format: 0,
            flags: 0,
            orig_size: 4096,
            name: "model.weight".to_string(),
            payload_hash: None,
            inline_chain: None,
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![identity_plane()],
        };
        let mut out = Vec::new();
        write_tensor_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_tensor_record(&out).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(parsed.chain_ref, 0x0003);
        assert!(parsed.inline_chain.is_none());
        assert_eq!(parsed.name, "model.weight");
    }

    #[test]
    fn tensor_record_with_inline_chain_roundtrip() {
        // chain_ref = 0x8001 (high bit set), inline_chain populated.
        let chain = minimal_chain();
        let rec = TensorRecord {
            chain_ref: 0x8001,
            dtype_code: 0x0010,
            input_format: 0,
            flags: 0,
            orig_size: 2048,
            name: "lm_head.weight".to_string(),
            payload_hash: Some(0xDEAD_BEEF_CAFE_1234),
            inline_chain: Some(chain.clone()),
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![identity_plane(), identity_plane()],
        };
        let mut out = Vec::new();
        write_tensor_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_tensor_record(&out).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(parsed.chain_ref, 0x8001);
        assert!(parsed.chain_ref & CHAIN_REF_INLINE_BIT != 0);
        assert_eq!(parsed.inline_chain, Some(chain));
        assert_eq!(parsed.name, "lm_head.weight");
        assert_eq!(parsed.payload_hash, Some(0xDEAD_BEEF_CAFE_1234));
        assert_eq!(parsed.terminals.len(), 2);
    }

    #[test]
    fn record_with_deps_and_hash_roundtrip() {
        let mut expected_hash = [0u8; 32];
        expected_hash[0] = 0xAB;
        expected_hash[31] = 0xCD;

        let rec = TensorRecord {
            chain_ref: 0x0002,
            dtype_code: 0x0002,
            input_format: 0,
            flags: 0,
            orig_size: 4096,
            name: "model.layers.0.weight".to_string(),
            payload_hash: Some(0xDEAD_BEEF_CAFE_1234),
            inline_chain: None,
            dependencies: vec![Dependency {
                ref_kind: RefKind::LocalTensor,
                expected_hash: Some(expected_hash),
                ref_bytes: b"model.layers.0.scale".to_vec(),
            }],
            tensor_metadata: None,
            terminals: vec![identity_plane()],
        };
        let mut out = Vec::new();
        write_tensor_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_tensor_record(&out).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(parsed.chain_ref, rec.chain_ref);
        assert_eq!(parsed.name, rec.name);
        assert_eq!(parsed.payload_hash, rec.payload_hash);
        assert_eq!(parsed.dependencies.len(), 1);
        assert_eq!(parsed.dependencies[0].ref_kind, RefKind::LocalTensor);
        assert_eq!(parsed.dependencies[0].expected_hash, Some(expected_hash));
        assert_eq!(parsed.dependencies[0].ref_bytes, b"model.layers.0.scale");
        assert_eq!(parsed.terminals, rec.terminals);
    }

    #[test]
    fn record_with_metadata_roundtrip() {
        let cbor_bytes = b"\xa1\x63key\x63val".to_vec();
        let rec = TensorRecord {
            chain_ref: 0x0001,
            dtype_code: 0x0001,
            input_format: 1,
            flags: 0,
            orig_size: 512,
            name: "emb".to_string(),
            payload_hash: None,
            inline_chain: None,
            dependencies: Vec::new(),
            tensor_metadata: Some(cbor_bytes.clone()),
            terminals: vec![identity_plane()],
        };
        let mut out = Vec::new();
        write_tensor_record(&rec, &mut out).unwrap();
        let (parsed, consumed) = parse_tensor_record(&out).unwrap();
        assert_eq!(consumed, out.len());
        assert_eq!(parsed.name, rec.name);
        assert_eq!(parsed.tensor_metadata, Some(cbor_bytes));
        assert_eq!(parsed.terminals, rec.terminals);
    }

    #[test]
    fn inline_bit_set_but_chain_none_rejected() {
        let rec = TensorRecord {
            chain_ref: 0x8000, // inline bit set
            dtype_code: 0,
            input_format: 0,
            flags: 0,
            orig_size: 0,
            name: "x".to_string(),
            payload_hash: None,
            inline_chain: None, // contradicts the inline bit
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![],
        };
        let mut out = Vec::new();
        assert!(write_tensor_record(&rec, &mut out).is_err());
    }

    #[test]
    fn inline_chain_some_but_bit_clear_rejected() {
        let rec = TensorRecord {
            chain_ref: 0x0000, // no inline bit
            dtype_code: 0,
            input_format: 0,
            flags: 0,
            orig_size: 0,
            name: "x".to_string(),
            payload_hash: None,
            inline_chain: Some(minimal_chain()), // contradicts the clear bit
            dependencies: Vec::new(),
            tensor_metadata: None,
            terminals: vec![],
        };
        let mut out = Vec::new();
        assert!(write_tensor_record(&rec, &mut out).is_err());
    }
}
