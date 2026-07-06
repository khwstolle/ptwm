---
title: Burrows-Wheeler
description: Block-wise Burrows-Wheeler Transform (BWT).
---

# Burrows-Wheeler Transform

The `BurrowsWheeler` transform applies a block-wise Burrows-Wheeler Transform (BWT) to cluster bytes that share similar contexts into long consecutive runs.

## Theory

The BWT is a reversible permutation of a byte block. While it provides no compression on its own, it acts as a powerful preprocessing stage. 

After PTWM's bit-reorder and byte-split chain, the exponent plane typically consists of long runs of similar bytes. BWT groups these identical bytes together. When followed by a [Move to Front](move-to-front) transform and an entropy coder, this pipeline (the classic `bzip2` approach) extracts high-order structural redundancy that block-local entropy coders cannot see.

Encoding uses prefix-doubling rotation sort `O(n log² n)`, while decoding uses a fast LF-mapping walk `O(n)`. This asymmetry is intentional, targeting release-time compression where slow encode is acceptable but fast decode is critical.

## Usage

The transform operates strictly on `Byte`-width planes and processes data in fixed 256 KiB blocks. Each output block is prefixed with a 4-byte little-endian primary index header.

## References

* Michael Burrows and David J. Wheeler, "A Block-sorting Lossless Data Compression Algorithm", *SRC Research Report 124* (1994). [Link](https://hplabs.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf)
