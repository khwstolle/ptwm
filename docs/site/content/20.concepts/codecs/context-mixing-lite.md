---
title: Context Mixing Lite
description: PAQ-style context mixing entropy coder.
---

# Context Mixing Lite

The `ContextMixingLite` codec provides a PAQ-style context mixing entropy coder for extreme compression.

## Theory

Context mixing coders use multiple independent models (e.g., different historical context lengths or patterns) to predict the next bit or byte. A mixer (often a neural network or logistic regression) combines these predictions dynamically based on which models have been most accurate recently. 

This approach achieves state-of-the-art compression ratios but is computationally expensive during both encoding and decoding.

## Usage

This codec is opt-in and is **not** included in the default trial-encode dispatch menu. It is only selected when explicitly named in the `codec_menu`. It requires a minimum plane size of 4 KiB to amortize its heavy startup costs.

## References

* Matt Mahoney, "Data Compression Using Neural Networks" / PAQ algorithm family. [Link](http://mattmahoney.net/dc/)
