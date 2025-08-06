/// Encodes the weight table into `out`.
///
/// Header format:
/// - 1-byte: `0x81..=0xFF` → n_symbols = `header & 0x7F` (1..=127)
/// - 2-byte: `0x80, n` → n_symbols = `n` (128..=256); `n == 0` encodes 256
///
/// Weights are nibble-packed (two per byte, low nibble first).
pub(crate) fn encode_weights(weights: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
    let n = weights.len();
    if n == 0 {
        return Err("empty weights".to_string());
    }
    if n > 256 {
        return Err(format!("too many symbols in weights: {n}"));
    }

    if n <= 127 {
        out.push(0x80 | (n as u8));
    } else {
        // Extended 2-byte header: escape 0x80 followed by count.
        // n=256 wraps to 0u8 by convention (the only value 0 can mean here).
        out.push(0x80);
        out.push(n as u8);
    }

    for chunk in weights.chunks(2) {
        let low = chunk[0] & 0x0F;
        let high = chunk.get(1).copied().unwrap_or(0) & 0x0F;
        out.push(low | (high << 4));
    }
    Ok(())
}

pub(crate) fn decode_weights(src: &[u8]) -> Result<(Vec<u8>, usize), String> {
    if src.is_empty() {
        return Err("missing weight header".to_string());
    }
    let header = src[0];
    if (header & 0x80) == 0 {
        return Err(format!("invalid weight header byte: {header:#04x}"));
    }

    let (n_symbols, header_bytes) = if header == 0x80 {
        // Extended 2-byte header: count byte of 0 encodes 256 symbols.
        if src.len() < 2 {
            return Err("truncated extended weight header".to_string());
        }
        let count_byte = src[1] as usize;
        let n = if count_byte == 0 { 256 } else { count_byte };
        if n < 128 {
            return Err(format!(
                "invalid extended weight count {n}: expected 128..=256"
            ));
        }
        (n, 2usize)
    } else {
        let n = (header & 0x7F) as usize;
        if n == 0 {
            return Err("zero weight symbols".to_string());
        }
        (n, 1usize)
    };

    let nibble_bytes = n_symbols.div_ceil(2);
    let needed = header_bytes + nibble_bytes;
    if src.len() < needed {
        return Err(format!(
            "weight payload truncated: need {needed}, got {}",
            src.len()
        ));
    }

    let mut weights = vec![0u8; n_symbols];
    let payload = &src[header_bytes..];
    for i in 0..n_symbols {
        let b = payload[i / 2];
        weights[i] = if i % 2 == 0 {
            b & 0x0F
        } else {
            (b >> 4) & 0x0F
        };
    }

    Ok((weights, needed))
}

#[cfg(test)]
mod tests {
    use super::{decode_weights, encode_weights};

    #[test]
    fn weights_roundtrip() {
        let w = vec![1, 2, 3, 4, 5];
        let mut out = Vec::new();
        encode_weights(&w, &mut out).unwrap();
        let (decoded, used) = decode_weights(&out).unwrap();
        assert_eq!(used, out.len());
        assert_eq!(decoded, w);
    }

    #[test]
    fn weights_roundtrip_127_symbols() {
        // Max count for 1-byte header path.
        let w: Vec<u8> = (0..127).map(|i| (i % 11) as u8 + 1).collect();
        let mut out = Vec::new();
        encode_weights(&w, &mut out).unwrap();
        assert_eq!(out[0], 0x80 | 127); // 1-byte header
        let (decoded, used) = decode_weights(&out).unwrap();
        assert_eq!(used, out.len());
        assert_eq!(decoded, w);
    }

    #[test]
    fn weights_roundtrip_128_symbols() {
        // Min count for 2-byte (extended) header path.
        let w: Vec<u8> = (0..128).map(|i| (i % 11) as u8 + 1).collect();
        let mut out = Vec::new();
        encode_weights(&w, &mut out).unwrap();
        assert_eq!(out[0], 0x80); // escape byte
        assert_eq!(out[1], 128); // count byte
        let (decoded, used) = decode_weights(&out).unwrap();
        assert_eq!(used, out.len());
        assert_eq!(decoded, w);
    }

    #[test]
    fn weights_roundtrip_256_symbols() {
        // Max count: 256 symbols, encoded as count byte 0.
        let w: Vec<u8> = (0..256).map(|i| (i % 11) as u8 + 1).collect();
        let mut out = Vec::new();
        encode_weights(&w, &mut out).unwrap();
        assert_eq!(out[0], 0x80); // escape byte
        assert_eq!(out[1], 0); // 256 wraps to 0
        let (decoded, used) = decode_weights(&out).unwrap();
        assert_eq!(used, out.len());
        assert_eq!(decoded, w);
    }

    #[test]
    fn weights_odd_count_pads_nibble() {
        // Odd number of weights: last nibble byte has 0 in high nibble.
        let w = vec![3, 5, 7];
        let mut out = Vec::new();
        encode_weights(&w, &mut out).unwrap();
        let (decoded, _) = decode_weights(&out).unwrap();
        assert_eq!(decoded, w);
    }

    #[test]
    fn decode_rejects_invalid_header() {
        // Byte without high bit set is invalid.
        assert!(decode_weights(&[0x42]).is_err());
    }

    #[test]
    fn decode_rejects_truncated_extended_header() {
        // 0x80 alone without count byte.
        assert!(decode_weights(&[0x80]).is_err());
    }

    #[test]
    fn encode_rejects_empty() {
        let mut out = Vec::new();
        assert!(encode_weights(&[], &mut out).is_err());
    }

    #[test]
    fn encode_rejects_too_many() {
        let w = vec![1u8; 257];
        let mut out = Vec::new();
        assert!(encode_weights(&w, &mut out).is_err());
    }
}
