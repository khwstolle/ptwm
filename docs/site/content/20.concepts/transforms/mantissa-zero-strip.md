---
title: Mantissa Zero Strip
description: Strips uniform trailing zero bits from quantized weight mantissas.
---

# Mantissa Zero Strip

The `MantissaZeroStrip` transform right-shifts every byte in a plane by `k` bits, safely removing uninformative trailing zeros.

## Theory

GPTQ and AWQ-style quantized checkpoints store weights with rounded mantissas. When you dequantize an INT4-quantized weight back into BF16 or FP16, the low bits of its mantissa are exactly zero. 

After PTWM runs the standard bit-reorder and byte-split chain, the resulting mantissa plane consists entirely of bytes sharing this uniform run of trailing zeros. Downstream entropy coders cannot automatically exploit this structure; they see an alphabet artificially inflated by a factor of 2^k, which wastes roughly `k` bits per element. 

By shifting the bytes right by `k`, `MantissaZeroStrip` shrinks the apparent alphabet, allowing the entropy coder to compress the plane efficiently. The inverse transform left-shifts the bytes by `k` to reconstruct the exact original values.

## Usage

The transform requires a single wire-format parameter `k` (between 1 and 7) representing the shift width. 

At encode time, the transform verifies that every input byte has at least `k` trailing zeros. If any byte contains non-zero data in those low bits, the transform returns an error and the trial-encoder skips this chain. This protects against information loss on unquantized data.

## References

* IEEE Standard for Floating-Point Arithmetic (IEEE 754).
