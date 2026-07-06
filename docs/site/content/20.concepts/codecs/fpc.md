---
title: FPC
description: Leading-zero byte compression for residual streams.
---

# FPC Codec

The `Fpc` codec is an implementation of a leading-zero plane codec, heavily inspired by floating-point compression (FPC) techniques like ndzip.

## Theory

When data is transformed through predictors (like the `PredictorXor` transform), the resulting residual streams often contain small-magnitude integers. These values are characterized by a large number of leading zero bytes.

The FPC codec efficiently encodes these values by storing a 4-bit "leading-zero-byte count" followed only by the significant (non-zero) suffix bytes. To minimize overhead, the 4-bit counts are packed two-per-byte in a header block before the suffix data.

## Usage

The codec targets byte planes containing packed element-wise residual streams. It is generally not evaluated by default but is activated when specific predictor chains are employed upstream in the Preprocessing Graph.

## References

* Martin Burtscher and Paruj Ratanaworabhan, "FPC: A High-Speed Compressor for Double-Precision Floating-Point Data", *IEEE Transactions on Computers* (2009). [Link](https://ieeexplore.ieee.org/document/4729544)
