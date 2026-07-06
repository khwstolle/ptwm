---
title: Neural Predictor
description: NNCP-style neural-network driven entropy coder.
---

# Neural Predictor

The `NeuralPredictor` codec is an opt-in NNCP-style (Neural Network Data Compression) entropy coder that uses a neural network to model byte probabilities.

## Theory

Unlike fixed-context or linear mixing models, this codec uses a deep neural network to predict the probability distribution of the next byte based on the previous bytes. The predicted distributions are then fed into a standard entropy coder (like arithmetic coding) to emit the compressed bitstream.

Neural predictors can capture complex, high-order correlations that traditional algorithms miss, yielding unprecedented compression ratios on complex data. However, this comes at the cost of significantly slower encoding and decoding speeds.

## Usage

Because of its high computational cost, this codec is **not** included in the default trial-encode dispatch menu. It must be explicitly selected via the `codec_menu`. It requires a minimum plane size of 4 KiB to amortize its per-bit overhead.

## References

* J. Schmidhuber & S. Heil, "Predictive Coding with Neural Nets: Application to Text Compression" (1996). [Link](https://people.idsia.ch/~juergen/tcs/tcs.html)
* Fabian Giesen, "Neural Network Data Compression Program (NNCP)".
