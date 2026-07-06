---
title: Zstd Dictionary
description: Zstandard dictionary-based compression.
---

# Zstd Dictionary

The `ZstdDict` codec is a variant of the standard Zstandard codec that supports pre-trained dictionaries.

## Theory

When the codec's `state_bytes` are populated, the bytes are treated as a pre-trained Zstd dictionary. This dictionary seeds the compressor's Lempel-Ziv window, allowing repeated structural patterns (such as attention block headers, embedding row layouts, or common tensor structures) to be compressed into much shorter back-references right from the beginning of the stream.

If the dictionary is empty, it behaves exactly like the plain [Zstd](zstd) codec.

## Usage

The dispatcher provides the dictionary bytes to the codec from a shared state block in the container prelude. This prevents duplicating the same dictionary for every compressed plane, minimizing overhead.

## References

* Yann Collet, Zstandard. RFC 8878. [Link](https://datatracker.ietf.org/doc/html/rfc8878)
