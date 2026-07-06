---
title: Concat
description: Concatenates multiple input planes into a single byte plane.
---

# Concat

The `Concat` transform concatenates multiple input planes into a single unified byte plane.

## Theory

Certain codecs or downstream transforms operate more efficiently on a single, contiguous byte stream rather than fragmented planes. `Concat` fuses `N` input planes into one, summing their byte lengths. 

The inverse operation reliably splits the concatenated plane back into the original `N` sub-planes using the stored length parameters.

## Usage

The transform requires an `input_lens` parameter array that specifies the byte length of each input plane. All input planes must share the same element width. The output is always a flat `Byte` plane.
