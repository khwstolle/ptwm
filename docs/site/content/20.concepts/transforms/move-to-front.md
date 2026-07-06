---
title: Move to Front
description: Transforms recurring byte values into small indices to aid entropy coding.
---

# Move to Front

The `MoveToFront` (MTF) transform replaces each byte with its index in a continuously updated list, moving recently accessed bytes to the front (index 0).

## Theory

MTF maintains a list of all 256 byte values. When it processes a byte, it outputs its current index in the list, then moves that byte to the front. As a result, bytes that appear frequently in close proximity are mapped to very small indices (often 0). 

This transform pairs exceptionally well with the [Burrows-Wheeler Transform](burrows-wheeler). BWT clusters identical bytes into long runs. MTF converts those long runs of identical bytes into long runs of zeros. Downstream entropy coders like Huffman or rANS can then compress these zero-runs extremely efficiently.

## Usage

The transform operates on raw byte streams and preserves the exact length of the input. It requires no parameters.

## References

* J. L. Bentley, D. D. Sleator, R. E. Tarjan, and V. K. Wei, "A locally adaptive data compression scheme", *Communications of the ACM* (1986). [Link](https://dl.acm.org/doi/10.1145/5684.5688)
