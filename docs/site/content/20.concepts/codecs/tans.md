---
title: tANS
description: Tabled Asymmetric Numeral Systems (Finite State Entropy) coder.
---

# tANS Codec

The `Tans` codec implements tabled Asymmetric Numeral Systems (also known as Finite State Entropy or FSE). 

## Theory

tANS provides the near-optimal compression ratios of arithmetic coding with the speed of a table-lookup decoder (similar to Huffman). It models symbol probabilities into a finite state machine where state transitions consume fractional bits.

Like [rANS](rans), it uses length-prefix framing and supports a `TAG_RAW` raw-fallback mechanism. If the data is incompressible, or if the tANS headers would expand the payload, it writes the original bytes directly.

## Usage

The codec is a peer of `rANS` and `Huffman` and is automatically trialed by the compressor's dispatcher to find the optimal coder for a given plane.

## References

* Jarek Duda, "Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding" (2013). [Link](https://arxiv.org/abs/1311.2540)
