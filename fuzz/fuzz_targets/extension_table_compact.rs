#![no_main]

//! Compact Extension Table parser. A table that parses must re-encode to
//! the same bytes: each table has one encoding.

use libfuzzer_sys::fuzz_target;
use ptwm_core::extension::compact_table::{compact_table_bytes, read_compact_table};

fuzz_target!(|data: &[u8]| {
    if let Ok(table) = read_compact_table(data) {
        let out = compact_table_bytes(&table).expect("a parsed table re-encodes");
        assert_eq!(out, data, "parsed table re-encodes to other bytes");
    }
});
