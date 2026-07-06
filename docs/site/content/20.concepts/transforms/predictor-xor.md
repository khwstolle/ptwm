---
title: Predictor XOR
description: FPC-style intra-plane predictor with an XOR residual.
---

# Predictor XOR

The `PredictorXor` transform uses a fixed-context model (FCM) to predict elements within a plane, emitting an XOR residual that entropy coders can compress efficiently.

## Theory

For each element in a plane, `PredictorXor` predicts its value from previous values using a small fixed-context hash table. It then emits `truth XOR prediction`.

When the underlying values change smoothly, the output residuals cluster around zero, exhibiting a heavy leading-zero distribution. Entropy coders like Huffman or FPC excel at compressing these distributions. 

The predictor maintains a 64K-entry hash table indexed by a rolling hash of recent values. The decoder runs the exact same recurrence in lock-step to recover the original sequence bit-exactly.

## Usage

The transform operates on `Byte`, `Word2`, `Word4`, and `Word8` element widths. It requires no parameters, as the hash table size and shift parameters are fixed.

## References

* Martin Burtscher and Paruj Ratanaworabhan, "FPC: A High-Speed Compressor for Double-Precision Floating-Point Data", *IEEE Transactions on Computers* (2009). [Link](https://ieeexplore.ieee.org/document/4729544)
