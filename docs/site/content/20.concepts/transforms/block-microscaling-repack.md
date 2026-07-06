---
title: Block Microscaling Repack
description: Re-expresses a 16-bit float plane as a per-block shared scale and a scale-relative element plane.
---

# Block Microscaling Repack

The `BlockMicroscalingRepack` transform losslessly converts a 16-bit float plane into a per-block shared scale (E8M0 format) and a scale-relative element plane.

## Theory

The OCP microscaling formats (MXFP4 / NVFP4) store one shared block scale alongside their elements. PTWM typically compresses that scale stream very well because the E8M0 sequence is smooth and autocorrelated. 

However, standard BF16 weights that aren't stored in a microscaled container leave this headroom on the table. `BlockMicroscalingRepack` recovers this headroom without any precision loss. 

The transform splits a block of consecutive elements into:
1. **Scale**: The maximum 8-bit exponent field in the block.
2. **Elements**: The exponent field of each element is replaced by the non-negative residual `scale - exponent`. The sign and mantissa bits remain untouched.

Because the scale is the block maximum, every residual falls between 0 and 255, and reconstruction is exact. The compression benefit comes from the new residual exponent field concentrating near zero, which makes the downstream exponent plane far more compressible.

## Usage

The transform requires a `block_size` parameter representing the number of elements per block. It produces two output planes:
- **Element Plane**: Same shape and element width (`Word2`) as the input, with exponent fields rebased to residuals.
- **Scale Plane**: One E8M0 byte per block.

## References

* OCP Microscaling Formats (MX) Specification v1.0. [Link](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-pdf)
