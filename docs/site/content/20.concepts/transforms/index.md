---
title: Transforms
description: A complete catalogue of PTWM Preprocessing Graph nodes.
---

# Transforms

PTWM's Preprocessing Graph (PPG) routes every tensor through a typed DAG of preprocessing operations, known as **transforms**. 

Each transform alters the structure or layout of the tensor's bytes to prepare them for entropy coding, exposing skewed distributions or aligning bitwise structures (like IEEE 754 exponents) onto byte or nibble boundaries.

## Available Transforms

- [Alpha-Stable Normalize](alpha-stable-normalize)
- [Bit Reorder](bit-reorder)
- [Block Microscaling Repack](block-microscaling-repack)
- [Burrows-Wheeler](burrows-wheeler)
- [Byte Split](byte-split)
- [Concat](concat)
- [Delta](delta)
- [Index Bitwidth Pack](index-bitwidth-pack)
- [IntDelta](int-delta)
- [Mantissa Zero Strip](mantissa-zero-strip)
- [Move to Front](move-to-front)
- [MxFp4 Deinterleave](mxfp4-deinterleave)
- [Predictor XOR](predictor-xor)
- [Reshape](reshape)
- [Spherical Normalize](spherical-normalize)
