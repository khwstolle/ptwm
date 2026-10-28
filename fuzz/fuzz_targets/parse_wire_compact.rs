#![no_main]

//! Compact (`\x89PTWC`) tensor record parser. A record that parses must
//! re-encode to the same bytes: each record has one encoding.

use libfuzzer_sys::fuzz_target;
use ptwm_core::codec::CodecId;
use ptwm_core::tensor_record_compact::{RecordContext, parse_compact_record, write_compact_record};

/// Extension Table entry 0 is a transform, 1 and 2 are plane codecs.
const CODEC_IDS: &[Option<CodecId>] = &[None, Some(CodecId::Zstd), Some(CodecId::Identity)];
/// Registry chain 0 is explicit, chain 1 has a bound Source.
const BOUND: &[bool] = &[false, true];

fuzz_target!(|data: &[u8]| {
    let ctx = RecordContext {
        codec_ids: CODEC_IDS,
        registry_bound: BOUND,
    };
    if let Ok(rec) = parse_compact_record(data, &ctx) {
        let mut out = Vec::new();
        write_compact_record(&rec, &ctx, &mut out).expect("a parsed record re-encodes");
        assert_eq!(out, data, "parsed record re-encodes to other bytes");
    }
});
