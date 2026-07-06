---
title: Concepts
description: Architecture and design rationale for PTWM's compression pipeline.
---

A few architectural decisions shape PTWM. This section walks through each,
in the order a tensor encounters them during compression.

| Concept | One-line summary |
|---|---|
| [Preprocessing Graph (PPG)](/concepts/ppg) | Routes every tensor through a typed DAG of ops before entropy coding. |
| [Transforms](/concepts/transforms) | Individual preprocessing operations (e.g., bit reorder, byte split, delta). |
| [Plane decomposition](/concepts/planes) | Bit reorder + byte / nibble split isolates the exponent into its own plane. |
| [Codecs](/concepts/codecs) | Huffman, rANS, Identity, Zstd, Order-1 ScaleAC — chosen per-plane by trial encoding. |
| [`.ptwm` container](/concepts/container) | Multi-tensor file with random-access lookup, hash-verified payloads, codec dispatch. |
| [Delta compression](/concepts/delta) | XOR-residuals against a reference frame for checkpoint chains. |
| [Extensibility](/concepts/extensibility) | The 13 contribution kinds × 3 flavors that let new codecs and transforms ship as signed bundles. |

## Why bit reordering matters

IEEE 754 floats place the sign bit between the exponent and mantissa,
straddling a byte boundary. Without a bit reorder, the sign contaminates
the most-significant byte and the exponent looks near-random. The
[preprocessing graph](/concepts/ppg) moves the exponent into its own byte
(or nibble for FP8); that byte then compresses to about 2.6 bits/byte
entropy on real-world trained weights.

For microscaling formats (MXFP4 / NVFP4) the situation flips: the nibble
plane is already near-uniform, and the lossless headroom sits in the
**scale** plane. PTWM ships dedicated codecs for E8M0 and FP8-E4M3 scale
distributions; see [Codecs](/concepts/codecs).
