---
title: Huffman LLM 5-Bit
description: Field-separated Huffman codec for 16-bit float planes.
---

# Huffman LLM 5-Bit

The `HuffLlm5Bit` codec is a specialized Huffman entropy coder designed for 16-bit floating-point planes (FP16 or BF16) using a field-separated bit layout.

## Theory

Instead of treating a 16-bit float simply as two independent 8-bit bytes, this codec leverages the internal structure of the floating-point representation. It uses a bit-field decomposition (such as the `{1, 5, 5, 5}` layout for FP16 or `{1, 4, 4, 7}` for BF16) and trial-encodes both to find the most efficient representation. 

By separating the fields, the Huffman models can compress the highly correlated bits (like the sign and exponent fields) independently from the noisier mantissa bits, yielding better compression for unaltered 16-bit weights. 

## Usage

This codec operates on `Role::Raw` 16-bit planes. It is self-delimiting (the word count is framed in the payload) and includes a raw fallback mechanism to gracefully handle incompressible data.

## References

* This codec is an original adaptation of Huffman coding designed specifically for LLM weight distributions in PTWM.
