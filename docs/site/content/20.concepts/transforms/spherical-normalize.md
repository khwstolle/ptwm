---
title: Spherical Normalize
description: Per-row unit-vector reparameterization with an exact XOR residual.
---

# Spherical Normalize

The `SphericalNormalize` transform performs a per-row (radius, direction) reparameterization on 2D float tensors, capturing row-wise correlations while ensuring a bit-exact round-trip via an XOR residual.

## Theory

Trained weight matrices often exhibit strong row-wise correlation. `SphericalNormalize` exploits this by representing each row as a magnitude (radius) and a unit vector (direction). 

The transform splits the input into three planes:
1. **Radius**: A single `f32` value per row, representing the Euclidean norm (`L2`).
2. **Direction**: The unit vector of the row, stored as `bf16` elements.
3. **Residual**: A bit-exact XOR difference between the original float bytes and the reconstructed `radius * direction` prediction bytes.

This approach is lossless by construction. Float-op inaccuracy only affects the size of the residual, never the correctness of the final output. The `bf16` quantized direction vectors compress well, and the XOR residual is concentrated around zero.

## Usage

The transform requires the input to have a `Rows` layout.

## References

* This bit-exact spherical reparameterization (radius and quantized direction vector with exact XOR residual) is an original transform introduced in PTWM for capturing row-wise correlations in deep learning weights.
