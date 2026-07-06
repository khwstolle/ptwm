---
title: Zstd
description: Zstandard generic dictionary compression.
---

# Zstd Codec

The `Zstd` codec wraps the standard Zstandard algorithm, allowing the container to use it as a generic meta-compressor.

## Theory

Zstandard provides excellent generic compression. In PTWM, it is primarily used as a fallback or forced via the `Method::ZSTD` selection. 

Because the upstream byte-split preprocessing ops already perform the SHUFFLE-style reordering that Zstandard would typically need, the plane codec simply runs plain Zstd over the preprocessed data. PTWM uses a default compression level of 3, which offers a strong balance between speed and ratio.

## Usage

This codec accepts any plane descriptor.

## References

* Yann Collet, Zstandard. RFC 8878. [Link](https://datatracker.ietf.org/doc/html/rfc8878)
