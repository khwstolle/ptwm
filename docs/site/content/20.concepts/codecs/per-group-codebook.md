---
title: Per-Group Codebook
description: Models per-group nibble distributions for quantized MXFP4 value planes.
---

# Per-Group Codebook

The `PerGroupCodebook` codec models the per-group nibble distribution of quantized MXFP4 `VALUE` planes.

## Theory

In microscaled formats like MXFP4, weights are grouped into blocks (typically 32 elements). The value distributions across different blocks can vary significantly. 

This codec extracts a histogram of nibble occurrences for every group in the plane. It then runs a multi-seed, deterministic k-means clustering algorithm to find a set of `K` representative PMFs (the codebook centroids). 

Each group is assigned to its nearest centroid. The codec range-codes the nibbles in each group using the assigned centroid's PMF, heavily compressing groups that have highly skewed local distributions.

## Usage

This codec exclusively accepts nibble-packed `Value` planes with a `Nibble` element width.

## References

* Robert M. Gray, "Vector Quantization", *IEEE ASSP Magazine* (1984). [Link](https://ieeexplore.ieee.org/document/1162229)
