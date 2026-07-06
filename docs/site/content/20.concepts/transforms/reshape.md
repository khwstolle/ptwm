---
title: Reshape
description: Re-tags a plane's layout without modifying its bytes.
---

# Reshape

The `Reshape` transform modifies the `Layout` metadata of a plane without altering the underlying byte array.

## Theory

Certain downstream codecs (like the row-aware [Order-1 Scale AC](../codecs/order1-scale-ac)) require the plane to have a specific shape, such as a `Rows` layout with a particular `row_len`. 

The `Reshape` operation converts a `Flat` layout into a structured `Rows` layout (provided the total length is perfectly divisible by the row length), or conversely collapses a `Rows` layout back into a `Flat` one.

## Inverse

Because no bytes are actually moved or modified, the inverse of `Reshape` is a pass-through of the bytes, with the original layout context restored automatically by the preceding node in the inverse Preprocessing Graph.
