//! Minimal standard-alphabet base64 (RFC 4648) with padding.
//!
//! Kept dependency-free: the transcoder only needs to round-trip small opaque
//! byte fields (op params, edge vendor bytes, codec state). Standard alphabet,
//! `=` padding, no line wrapping.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `data` as a standard-alphabet, padded base64 string.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn decode_char(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a' + 26) as u32),
        b'0'..=b'9' => Some((c - b'0' + 52) as u32),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Decode a standard-alphabet, padded base64 string. Returns `None` on any
/// malformed input (bad characters, wrong length, misplaced padding).
pub fn decode(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let num_chunks = bytes.len() / 4;
    for (chunk_idx, chunk) in bytes.chunks(4).enumerate() {
        let mut n = 0u32;
        let mut pad = 0usize;
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                // Padding may only appear in the final positions of the final
                // chunk.
                if i < 2 || chunk_idx + 1 != num_chunks {
                    return None;
                }
                pad += 1;
                n <<= 6;
            } else {
                let v = decode_char(c)?;
                if pad > 0 {
                    return None; // non-padding after padding
                }
                n = (n << 6) | v;
            }
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_lengths() {
        for len in 0..=32usize {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            let enc = encode(&data);
            let dec = decode(&enc).unwrap();
            assert_eq!(dec, data, "len={len}");
        }
    }

    #[test]
    fn known_vectors() {
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"f"), "Zg==");
        assert_eq!(encode(b"fo"), "Zm8=");
        assert_eq!(encode(b"foo"), "Zm9v");
        assert_eq!(encode(b"foob"), "Zm9vYg==");
        assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar");
    }

    #[test]
    fn rejects_malformed() {
        assert!(decode("Zg=").is_none()); // wrong length
        assert!(decode("Z===").is_none()); // padding too early
        assert!(decode("****").is_none()); // bad chars
        assert!(decode("Zm9=Yg==").is_none()); // data after padding within chunk
    }
}
