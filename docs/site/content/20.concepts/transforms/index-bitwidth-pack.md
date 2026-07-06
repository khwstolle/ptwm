---
title: Index Bitwidth Pack
description: Packs raw u8 indices into a tight MSB-first bit stream.
---

# Index Bitwidth Pack

The `IndexBitwidthPack` transform compresses arrays of indices that do not require a full 8 bits into a tightly packed bitstream.

## Theory

When representing cluster assignments or codebook indices (for example, downstream of a K-means quantization mechanism), the indices often belong to a small domain (e.g., `0..15` requiring only 4 bits). Storing them as full `u8` bytes wastes significant space.

This transform takes the `bits` parameter and packs the stream of indices MSB-first into a dense byte array, successfully recovering the bit budget.

## Inverse

The inverse operation unpacks the dense bitstream back into standard `u8` elements by extracting the appropriate `bits` intervals, yielding the original indices.
