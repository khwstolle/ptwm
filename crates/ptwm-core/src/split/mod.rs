//! Byte/bit preprocessing for neural network weight compression.
//!
//! Neural network weights stored as IEEE 754 floats have a structure that
//! standard entropy coders (Huffman, ANS) cannot exploit directly: the sign,
//! exponent, and mantissa bits are interleaved across byte boundaries, making
//! every byte look near-random (~8 bits of entropy).
//!
//! This module applies two transformations before Huffman coding:
//!
//! ## Stage 1: Bit reorder (`bits_mode = 1`)
//!
//! Rearranges bits within each float so the exponent occupies a dedicated
//! byte position. For fp32:
//!
//! ```text
//! IEEE 754:  [S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM]  (sign, exponent, mantissa)
//! Reordered: [EEEEEEEE S MMMMMMMMMMMMMMMMMMMMMMM]  (exponent, sign, mantissa)
//! ```
//!
//! In little-endian memory, this moves the exponent into byte 3 (the MSB).
//! Without this step, the sign bit straddles the exponent/mantissa byte
//! boundary, contaminating both bytes and reducing compressibility.
//!
//! The same principle applies to fp16/bf16, where the exponent is shifted
//! into byte 1 of each 2-byte element.
//!
//! ## Stage 2: Byte split (round-robin deinterleaving)
//!
//! Distributes bytes across `num_buf` planes via `byte[i] → plane[i % num_buf]`.
//! For fp32 (`bytes_mode = 220`, `num_buf = 4`), this produces:
//!
//! - **Plane 0**: byte 0 of every float (mantissa low bits)
//! - **Plane 1**: byte 1 of every float (mantissa mid bits)
//! - **Plane 2**: byte 2 of every float (mantissa high + sign)
//! - **Plane 3**: byte 3 of every float (**exponent** — after bit reorder)
//!
//! Each plane is then Huffman-compressed independently. The exponent plane
//! compresses extremely well because trained weights cluster in a narrow
//! exponent range (typically ~20 unique values out of 256, giving ~2.6
//! bits/byte entropy vs ~8.0 for mantissa planes). This is where nearly
//! all compression savings come from.
//!
//! ## Why bit reorder matters
//!
//! The byte split (stage 2) alone is a standard byte-shuffle preprocessing
//! step, well known in scientific-array compression. The bit reorder
//! (stage 1) is what makes the exponent plane cleanly separable from the
//! sign bit. Without it, the sign bit bleeds across the byte 2/byte 3
//! boundary, increasing the exponent byte's entropy.

pub mod dtype16;
pub mod dtype32;
pub mod dtype64;
pub mod dtype8;
