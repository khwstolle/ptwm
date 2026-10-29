//! Unsigned LEB128 varints and a bounds-checked byte cursor for the compact
//! wire formats (the `\x89PTWS` frame and the compact chain wire).
//!
//! Parsing is canonical: a varint must use the fewest bytes that hold its
//! value (no trailing `0x80`/`0x00` groups), is at most 10 bytes long, and
//! must fit in `u64`, so each value has one varint encoding. The compact
//! formats build on this: each of their fields has one encoding for the
//! value it holds. (A compact frame as a whole still leaves the writer some
//! choices; see [`crate::compact`].)

use crate::error::PtwmCoreError;

/// Longest LEB128 encoding of a `u64`.
pub const MAX_VARINT_LEN: usize = 10;

/// Append `v` as an unsigned LEB128 varint.
pub fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Number of bytes [`write_varint`] uses for `v`.
pub fn varint_len(v: u64) -> usize {
    let bits = 64 - v.leading_zeros() as usize;
    bits.div_ceil(7).max(1)
}

/// Why a varint did not parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarintError {
    /// The buffer ended inside the varint.
    Truncated,
    /// The value does not fit in `u64` (more than 10 bytes, or a 10th byte
    /// above 1).
    Overflow,
    /// The varint has a redundant trailing zero group.
    NonMinimal,
}

impl VarintError {
    fn message(self) -> &'static str {
        match self {
            VarintError::Truncated => "truncated varint",
            VarintError::Overflow => "varint overflows u64",
            VarintError::NonMinimal => "non-minimal varint",
        }
    }
}

/// Read one canonical varint from `buf` at `*pos` and advance `*pos`.
pub fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, VarintError> {
    let mut v: u64 = 0;
    for i in 0..MAX_VARINT_LEN {
        let b = *buf.get(*pos + i).ok_or(VarintError::Truncated)?;
        if i == MAX_VARINT_LEN - 1 && b > 1 {
            return Err(VarintError::Overflow);
        }
        v |= u64::from(b & 0x7F) << (7 * i);
        if b & 0x80 == 0 {
            // A zero final group after the first byte adds nothing: the
            // value has a shorter encoding.
            if i > 0 && b == 0 {
                return Err(VarintError::NonMinimal);
            }
            *pos += i + 1;
            return Ok(v);
        }
    }
    Err(VarintError::Overflow)
}

/// A bounds-checked reader over one wire section. Every error is an
/// [`PtwmCoreError::InvalidContainer`] prefixed with the cursor's context
/// (for example `"compact frame"`).
pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    context: &'static str,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8], context: &'static str) -> Self {
        Cursor {
            buf,
            pos: 0,
            context,
        }
    }

    /// Error carrying this cursor's context.
    pub fn err(&self, msg: impl std::fmt::Display) -> PtwmCoreError {
        PtwmCoreError::InvalidContainer(format!("{}: {msg}", self.context))
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Move to an absolute position at or before the end of the buffer.
    pub fn set_pos(&mut self, pos: usize) {
        assert!(pos <= self.buf.len(), "cursor position past the end");
        self.pos = pos;
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn is_empty(&self) -> bool {
        self.pos == self.buf.len()
    }

    /// The unread bytes, without consuming them.
    pub fn rest(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], PtwmCoreError> {
        if n > self.remaining() {
            return Err(self.err("truncated"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn array<const N: usize>(&mut self) -> Result<[u8; N], PtwmCoreError> {
        Ok(self.take(N)?.try_into().expect("take returned N bytes"))
    }

    pub fn u8(&mut self) -> Result<u8, PtwmCoreError> {
        Ok(self.take(1)?[0])
    }

    pub fn u32_le(&mut self) -> Result<u32, PtwmCoreError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub fn u64_le(&mut self) -> Result<u64, PtwmCoreError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    /// One canonical varint.
    pub fn varint(&mut self) -> Result<u64, PtwmCoreError> {
        read_varint(self.buf, &mut self.pos).map_err(|e| self.err(e.message()))
    }

    /// A varint that must not exceed `max`; `what` names the field.
    pub fn varint_max(&mut self, max: u64, what: &str) -> Result<u64, PtwmCoreError> {
        let v = self.varint()?;
        if v > max {
            return Err(self.err(format_args!("{what} {v} exceeds {max}")));
        }
        Ok(v)
    }

    /// A varint used as a byte length: it must not exceed the bytes left.
    pub fn len_varint(&mut self) -> Result<usize, PtwmCoreError> {
        let v = self.varint()?;
        if v > self.remaining() as u64 {
            return Err(self.err("length exceeds remaining bytes"));
        }
        Ok(v as usize)
    }

    /// A length-prefixed byte string (`len_varint` + bytes).
    pub fn len_bytes(&mut self) -> Result<&'a [u8], PtwmCoreError> {
        let n = self.len_varint()?;
        self.take(n)
    }

    /// Fail unless every byte was consumed.
    pub fn finish(&self, what: &str) -> Result<(), PtwmCoreError> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(self.err(format_args!(
                "{} trailing bytes after {what}",
                self.remaining()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trips() {
        for v in [
            0u64,
            1,
            127,
            128,
            300,
            16383,
            16384,
            u32::MAX as u64,
            1 << 63,
            u64::MAX,
        ] {
            let mut out = Vec::new();
            write_varint(&mut out, v);
            assert_eq!(out.len(), varint_len(v), "{v}");
            let mut pos = 0;
            assert_eq!(read_varint(&out, &mut pos), Ok(v));
            assert_eq!(pos, out.len());
        }
        assert_eq!(varint_len(u64::MAX), MAX_VARINT_LEN);
    }

    #[test]
    fn non_minimal_varints_are_rejected() {
        for bytes in [
            &[0x80, 0x00][..],
            &[0x81, 0x00],
            &[0xFF, 0x80, 0x00],
            &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00],
        ] {
            let mut pos = 0;
            assert_eq!(
                read_varint(bytes, &mut pos),
                Err(VarintError::NonMinimal),
                "{bytes:02x?}"
            );
            assert_eq!(pos, 0);
        }
    }

    #[test]
    fn overflowing_and_truncated_varints_are_rejected() {
        let mut pos = 0;
        let eleven = [0xFFu8; 11];
        assert_eq!(read_varint(&eleven, &mut pos), Err(VarintError::Overflow));
        let mut ten = [0xFFu8; 10];
        ten[9] = 0x02;
        assert_eq!(read_varint(&ten, &mut pos), Err(VarintError::Overflow));
        ten[9] = 0x01;
        assert_eq!(read_varint(&ten, &mut pos), Ok(u64::MAX));
        let mut pos = 0;
        assert_eq!(read_varint(&[0x80], &mut pos), Err(VarintError::Truncated));
        assert_eq!(read_varint(&[], &mut pos), Err(VarintError::Truncated));
    }

    #[test]
    fn cursor_bounds_lengths_by_remaining_bytes() {
        let mut c = Cursor::new(&[3, 1, 2, 3, 9], "test");
        assert_eq!(c.len_bytes().unwrap(), &[1, 2, 3]);
        assert!(c.finish("x").is_err());
        assert_eq!(c.u8().unwrap(), 9);
        c.finish("x").unwrap();

        let mut c = Cursor::new(&[4, 1, 2, 3], "test");
        let err = c.len_varint().unwrap_err().to_string();
        assert!(err.contains("test: length exceeds"), "{err}");

        let mut c = Cursor::new(&[0x80, 0x02], "test");
        assert!(c.varint_max(255, "count").is_err());
    }
}
