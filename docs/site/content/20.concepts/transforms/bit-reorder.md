---
title: Bit Reorder
description: Rearranges IEEE-754 float bits to align the exponent onto a byte or nibble boundary.
---

# Bit Reorder

The `BitReorderIeee16`, `BitReorderIeee32`, `BitReorderFp8E4M3`, and `BitReorderFp8E5M2` transforms rearrange float bits so the exponent occupies the most-significant position.

## Theory

IEEE-754 floats place the sign bit on the exponent-mantissa byte boundary. This misaligned boundary makes raw float bytes look near-random, reducing the effectiveness of entropy coders. The bit-reorder transforms shift the exponent bits into a dedicated byte (or nibble for FP8), completely separating them from the sign and mantissa.

**Float32 Layout Example:**

| Byte | 3 (MSB) | 2 | 1 | 0 (LSB) |
| :--- | :---: | :---: | :---: | :---: |
| **IEEE 754** | `S EEEEEEE` | `E MMMMMMM` | `MMMMMMMM` | `MMMMMMMM` |
| **Reordered** | `EEEEEEEE` | `S MMMMMMM` | `MMMMMMMM` | `MMMMMMMM` |

After reordering, downstream transforms (like [Byte Split](byte-split)) can easily isolate the exponent plane. The exponent plane of trained weights typically carries very low entropy (roughly 2.6 bits/byte) because most weights cluster around a narrow range of magnitudes. Isolating this plane yields almost all of PTWM's compression savings.

## Usage

The transform operates in-place on the elements.

- **BitReorderIeee16**: Reorders 16-bit floats (BF16 or FP16). The input must have an even number of bytes.
- **BitReorderIeee32**: Reorders 32-bit floats (FP32). The input length must divide by four.
- **BitReorderFp8E4M3**: Reorders FP8-E4M3FN bytes from `[S EEEE MMM]` to `[EEEE SMMM]`.
- **BitReorderFp8E5M2**: Reorders FP8-E5M2 bytes from `[S EEEEE MM]` to `[EEEEE SMM]`.

## References

* Martin Burtscher and Paruj Ratanaworabhan, "FPC: A High-Speed Compressor for Double-Precision Floating-Point Data", *IEEE Transactions on Computers* (2009). [Link](https://ieeexplore.ieee.org/document/4729544)
