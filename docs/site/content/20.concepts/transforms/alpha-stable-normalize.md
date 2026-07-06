---
title: Alpha-Stable Normalize
description: Per-row robust affine normalization with an exact XOR residual.
---

# Alpha-Stable Normalize

The `AlphaStableNormalize` transform applies a per-row robust affine normalization `(v - δ) / γ` to 2D tensors, backed by an exact XOR residual to guarantee lossless reconstruction.

## Theory

This transform isolates heavy-tailed, row-wise scales and offsets. It calculates:
- `δ`: The median of the row.
- `γ`: The median absolute deviation (MAD) from `δ`.

By using the median and MAD instead of the mean and variance, the fit remains robust against extreme outliers. The transform splits the input into three planes:
1. **Scale**: The `(γ, δ)` pair for each row, stored as `f32`.
2. **Normalized**: The normalized elements `(v - δ) / γ`, quantized to `bf16`.
3. **Residual**: The bit-exact XOR difference between the original float bytes and the `bf16`-reconstructed bytes.

Because the `γ` and `δ` parameters are stored per row and the residual repairs all round-off errors, reconstruction is bit-exact regardless of the underlying distribution.

## Usage

The transform requires the input to have a `Rows` layout and supports `f32`, `bf16`, and `fp16` dtypes.

## References

* This robust affine normalization approach (using median and MAD with exact XOR residual) is an original contribution to PTWM, inspired by robust statistics for heavy-tailed distributions.
