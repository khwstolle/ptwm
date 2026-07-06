---
title: Byte Split & Nibble Split
description: Deinterleaves a flat byte plane into separate planes by bit significance.
---

# Byte Split and Nibble Split

The `ByteSplit` and `NibbleSplit` transforms deinterleave bytes or nibbles by significance to isolate predictable bit patterns into their own planes.

## Byte Split

`ByteSplit` performs a round-robin deinterleave of a flat byte plane into `n` separate planes (where `n` is 2, 4, or 8).

For an input stream splitting into 4 planes, the bytes are routed as follows:

<table style="text-align: center; font-family: monospace; border-collapse: collapse; width: 100%;">
<tr>
  <td style="border: 1px solid #ccc; padding: 4px;"><strong>Input</strong></td>
  <td style="border: 1px solid #ccc; padding: 4px;" colspan="4">Element 0</td>
  <td style="border: 1px solid #ccc; padding: 4px;" colspan="4">Element 1</td>
  <td style="border: 1px solid #ccc; padding: 4px;" colspan="4">Element 2</td>
</tr>
<tr>
  <td style="border: 1px solid #ccc; padding: 4px;"><strong>Bytes</strong></td>
  <td style="border: 1px solid #ccc; padding: 4px;">b0</td><td style="border: 1px solid #ccc; padding: 4px;">b1</td><td style="border: 1px solid #ccc; padding: 4px;">b2</td><td style="border: 1px solid #ccc; padding: 4px;">b3</td>
  <td style="border: 1px solid #ccc; padding: 4px;">b4</td><td style="border: 1px solid #ccc; padding: 4px;">b5</td><td style="border: 1px solid #ccc; padding: 4px;">b6</td><td style="border: 1px solid #ccc; padding: 4px;">b7</td>
  <td style="border: 1px solid #ccc; padding: 4px;">b8</td><td style="border: 1px solid #ccc; padding: 4px;">b9</td><td style="border: 1px solid #ccc; padding: 4px;">b10</td><td style="border: 1px solid #ccc; padding: 4px;">b11</td>
</tr>
<tr>
  <td style="border: 1px solid #ccc; padding: 4px;"><strong>Plane 0</strong></td>
  <td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(255, 99, 132, 0.2);">b0</td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td>
  <td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(255, 99, 132, 0.2);">b4</td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td>
  <td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(255, 99, 132, 0.2);">b8</td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td>
</tr>
<tr>
  <td style="border: 1px solid #ccc; padding: 4px;"><strong>Plane 1</strong></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(54, 162, 235, 0.2);">b1</td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(54, 162, 235, 0.2);">b5</td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(54, 162, 235, 0.2);">b9</td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td>
</tr>
<tr>
  <td style="border: 1px solid #ccc; padding: 4px;"><strong>Plane 2</strong></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(75, 192, 192, 0.2);">b2</td><td style="border: 1px solid #ccc; padding: 4px;"></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(75, 192, 192, 0.2);">b6</td><td style="border: 1px solid #ccc; padding: 4px;"></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(75, 192, 192, 0.2);">b10</td><td style="border: 1px solid #ccc; padding: 4px;"></td>
</tr>
<tr>
  <td style="border: 1px solid #ccc; padding: 4px;"><strong>Plane 3</strong></td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(255, 206, 86, 0.2);">b3</td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(255, 206, 86, 0.2);">b7</td>
  <td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px;"></td><td style="border: 1px solid #ccc; padding: 4px; background-color: rgba(255, 206, 86, 0.2);">b11</td>
</tr>
</table>

This transform typically follows a [Bit Reorder](bit-reorder) step. By grouping bytes of the same significance together (e.g., placing all the exponent bytes of a `float32` tensor into a single plane), the entropy coder sees highly skewed, compressible distributions rather than interleaved noise.

## Nibble Split

`NibbleSplit` performs the same isolation at a 4-bit granularity. It splits each input byte `[HHHH LLLL]` into two separate one-nibble-per-byte planes:
- **Plane 0 (High)**: The high nibble `(b >> 4) & 0x0F` (often the exponent for FP8 types).
- **Plane 1 (Low)**: The low nibble `b & 0x0F` (often the sign and mantissa).

This enables dtype-aware plane coding for sub-byte formats like FP8 and FP4.

## References

* Martin Burtscher and Paruj Ratanaworabhan, "FPC: A High-Speed Compressor for Double-Precision Floating-Point Data", *IEEE Transactions on Computers* (2009). [Link](https://ieeexplore.ieee.org/document/4729544)
