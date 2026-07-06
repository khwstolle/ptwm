---
title: Identity
description: A pass-through codec that performs no compression.
---

# Identity Codec

The `Identity` codec is a simple pass-through mechanism that stores planes as raw bytes without any compression.

## Theory

Certain planes may be mathematically incompressible (e.g., cryptographic nonces or highly dense random mantissas). The trial-encoding dispatcher evaluates the `Identity` codec alongside `Huffman` and `rANS`. If the entropy coders fail to compress the data (or if they expand it due to header overhead), the dispatcher selects the `Identity` codec to ensure the payload is never larger than the original raw bytes.

## Usage

This codec accepts any plane descriptor. It writes a zero-byte state and directly copies the payload.
