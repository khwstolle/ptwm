//! v1 PTWM wire-format header.
//!
//! Layout (little-endian):
//!
//! ```text
//!     magic:                  [u8; 9]      \x89PTWM\r\n\x1a\n or \x89PTWX\r\n\x1a\n
//!     flags:                  u16          bit 0 = .ptwx (has embedded WASM)
//!     extension_table_offset: u64          0 ⇒ no extensions referenced
//!     extension_table_length: u64
//!     extension_table_hash:   [u8; 32]     blake3 of serialized table
//!     tensor_index_offset:    u64
//!     tensor_index_length:    u64
//!     shared_prelude_offset:  u64
//!     shared_prelude_length:  u64
//! ```
//!
//! Fixed size: 9 + 2 + 8 + 8 + 32 + 8 + 8 + 8 + 8 = 91 bytes.

use std::io::{Cursor, Read, Write};

use crate::error::PtwmCoreError;

pub const MAGIC_PTWM: [u8; 9] = [0x89, b'P', b'T', b'W', b'M', b'\r', b'\n', 0x1A, b'\n'];
pub const MAGIC_PTWX: [u8; 9] = [0x89, b'P', b'T', b'W', b'X', b'\r', b'\n', 0x1A, b'\n'];

pub const FLAG_PTWX: u16 = 0x0001;

pub const HEADER_LEN: usize = 91;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Header {
    pub magic: [u8; 9],
    pub flags: u16,
    pub extension_table_offset: u64,
    pub extension_table_length: u64,
    pub extension_table_hash: [u8; 32],
    pub tensor_index_offset: u64,
    pub tensor_index_length: u64,
    pub shared_prelude_offset: u64,
    pub shared_prelude_length: u64,
}

impl Header {
    pub fn new_ptwm() -> Self {
        Self::default_with_magic(MAGIC_PTWM)
    }

    pub fn new_ptwx() -> Self {
        let mut h = Self::default_with_magic(MAGIC_PTWX);
        h.flags |= FLAG_PTWX;
        h
    }

    fn default_with_magic(magic: [u8; 9]) -> Self {
        Self {
            magic,
            flags: 0,
            extension_table_offset: 0,
            extension_table_length: 0,
            extension_table_hash: [0u8; 32],
            tensor_index_offset: 0,
            tensor_index_length: 0,
            shared_prelude_offset: 0,
            shared_prelude_length: 0,
        }
    }

    pub fn is_ptwx(&self) -> bool {
        self.flags & FLAG_PTWX != 0
    }

    pub fn write<W: Write>(&self, w: &mut W) -> Result<(), PtwmCoreError> {
        w.write_all(&self.magic)?;
        w.write_all(&self.flags.to_le_bytes())?;
        w.write_all(&self.extension_table_offset.to_le_bytes())?;
        w.write_all(&self.extension_table_length.to_le_bytes())?;
        w.write_all(&self.extension_table_hash)?;
        w.write_all(&self.tensor_index_offset.to_le_bytes())?;
        w.write_all(&self.tensor_index_length.to_le_bytes())?;
        w.write_all(&self.shared_prelude_offset.to_le_bytes())?;
        w.write_all(&self.shared_prelude_length.to_le_bytes())?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> Result<Self, PtwmCoreError> {
        let mut magic = [0u8; 9];
        r.read_exact(&mut magic)?;
        if magic != MAGIC_PTWM && magic != MAGIC_PTWX {
            return Err(PtwmCoreError::InvalidHeaderField(format!(
                "bad magic: {:02x?}",
                magic
            )));
        }
        let mut buf2 = [0u8; 2];
        r.read_exact(&mut buf2)?;
        let flags = u16::from_le_bytes(buf2);

        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let extension_table_offset = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let extension_table_length = u64::from_le_bytes(buf8);

        let mut extension_table_hash = [0u8; 32];
        r.read_exact(&mut extension_table_hash)?;

        r.read_exact(&mut buf8)?;
        let tensor_index_offset = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let tensor_index_length = u64::from_le_bytes(buf8);

        r.read_exact(&mut buf8)?;
        let shared_prelude_offset = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?;
        let shared_prelude_length = u64::from_le_bytes(buf8);

        Ok(Self {
            magic,
            flags,
            extension_table_offset,
            extension_table_length,
            extension_table_hash,
            tensor_index_offset,
            tensor_index_length,
            shared_prelude_offset,
            shared_prelude_length,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, PtwmCoreError> {
        let mut buf = Vec::with_capacity(HEADER_LEN);
        self.write(&mut buf)?;
        debug_assert_eq!(buf.len(), HEADER_LEN);
        Ok(buf)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PtwmCoreError> {
        if bytes.len() < HEADER_LEN {
            return Err(PtwmCoreError::InvalidHeaderField(format!(
                "header truncated: need {} bytes, got {}",
                HEADER_LEN,
                bytes.len()
            )));
        }
        let mut c = Cursor::new(bytes);
        Self::read(&mut c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptwm_roundtrip() {
        let h = Header::new_ptwm();
        let bytes = h.to_bytes().unwrap();
        assert_eq!(bytes.len(), HEADER_LEN);
        let back = Header::from_bytes(&bytes).unwrap();
        assert_eq!(back, h);
        assert!(!back.is_ptwx());
    }

    #[test]
    fn ptwx_roundtrip() {
        let h = Header::new_ptwx();
        let bytes = h.to_bytes().unwrap();
        let back = Header::from_bytes(&bytes).unwrap();
        assert_eq!(back, h);
        assert!(back.is_ptwx());
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut bytes = Header::new_ptwm().to_bytes().unwrap();
        bytes[1] = b'Z'; // corrupt 'P'
        let res = Header::from_bytes(&bytes);
        assert!(res.is_err());
    }

    #[test]
    fn rejects_truncated() {
        let bytes = Header::new_ptwm().to_bytes().unwrap();
        for cut in 0..HEADER_LEN {
            assert!(Header::from_bytes(&bytes[..cut]).is_err());
        }
    }
}
