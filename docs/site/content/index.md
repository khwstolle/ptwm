---
seo:
  title: PTWM — lossless compression for PyTorch model weights
  description: Lossless compression for PyTorch weights via exponent-plane separation, microscaling-aware codecs, and a random-access container.
---

::u-page-hero
---
class: dark:bg-gradient-to-b from-neutral-900 to-neutral-950
orientation: horizontal
---
#title
Lossless compression for [PyTorch weights]{.text-primary}.

#description
PTWM is a lossless weight-compression library for PyTorch checkpoints. It exploits the byte-level structure of IEEE 754 floats and the layout of microscaling formats (MXFP4 / NVFP4) to recover compression beyond what generic byte-level codecs reach, with parallel encode and decode through a native Rust extension.

#links
:::u-button
---
to: /quickstart
size: xl
trailing-icon: i-lucide-arrow-right
---
Quickstart
:::

:::u-button
---
icon: i-simple-icons-github
color: neutral
variant: outline
size: xl
to: https://github.com/khwstolle/ptwm
target: _blank
---
GitHub
:::

#default
```python [example.py]
from ptwm import Compressor, Decompressor, CompressionConfig, Format
import torch

config = CompressionConfig(input_format=Format.TORCH)
compressor, decompressor = Compressor(config), Decompressor()

tensor = torch.randn(1024, 1024, dtype=torch.bfloat16)
restored = decompressor.decompress(compressor.compress(tensor))
assert torch.equal(tensor, restored)
```
::

::u-page-section
---
class: dark:bg-neutral-950
---
#title
Components

#description
Six pieces, each built around the byte-level structure of trained weights: a preprocessing graph that exposes low-entropy planes, codecs that operate on them, a self-describing container, and integrations for the standard loading paths.

#features
:::u-page-feature
---
icon: i-lucide-split
---
#title
Exponent-plane separation

#description
IEEE 754 bit-reorder plus byte / nibble split isolates the exponent into its own plane. The resulting plane carries ≈ 2.6 bits/byte of entropy on trained weights; the raw byte stream appears near-uniform.
:::

:::u-page-feature
---
icon: i-lucide-layers
---
#title
Microscaling-aware codecs

#description
An order-1 arithmetic coder fitted to MXFP4 E8M0 scale distributions compresses the scale plane to ≈ 0.16 of original. NVFP4 FP8-E4M3 scales route through an FP8-split chain.
:::

:::u-page-feature
---
icon: i-lucide-database
---
#title
Random-access container

#description
The `.ptwm` format bundles tensors with a name-indexed manifest, per-plane codec dispatch records, and hash-verified payloads. A reader fetches a single tensor without scanning the rest.
:::

:::u-page-feature
---
icon: i-lucide-zap
---
#title
Native Rust core

#description
PyO3 extension with rayon-parallel encode and decode, zero-copy buffer protocol, and pure-Rust Huffman and rANS codecs. No C dependencies.
:::

:::u-page-feature
---
icon: i-lucide-plug
---
#title
safetensors and HuggingFace integration

#description
The `patch_safetensors()` and `patch_transformers()` helpers decompress sibling `.ptwm` payloads inside the standard `load_state_dict` path; calling code requires no changes.
:::

:::u-page-feature
---
icon: i-lucide-terminal
---
#title
Command-line interface

#description
The `ptwm compress` and `ptwm decompress` commands accept single files, directories, and delta compression against a reference checkpoint.
:::
::

::u-page-section
---
class: dark:bg-neutral-950
ui:
  features: grid-cols-1 sm:grid-cols-2 lg:grid-cols-2
---
#title
Expected ratios

#description
Compressed bytes ÷ raw bytes — lower is better. Theoretical per-DType ranges; measured numbers across production checkpoints (Llama 3, Mixtral, GPT-OSS, Qwen, DiT) are reported on the [benchmarks page](/benchmarks).

#features
:::u-page-feature
---
icon: i-lucide-binary
---
#title
`bfloat16`, `float16`, `float32`

#description
≈ 0.55 – 0.75. Exponent-plane separation.
:::

:::u-page-feature
---
icon: i-lucide-hash
---
#title
`float8_e4m3fn`, `float8_e5m2`
#description
≈ 0.70 – 0.85. Nibble-split + plane-aware Huffman.
:::

:::u-page-feature
---
icon: i-lucide-grid-2x2
---
#title
MXFP4 (FP4 + E8M0)

#description
≈ 0.86 – 0.90. FP4 nibble plane near-uniform; the random-access container and scale-codec carry the gain.
:::

:::u-page-feature
---
icon: i-lucide-grid-2x2-plus
---
#title
NVFP4 (FP4 + FP8-E4M3)

#description
≈ 0.90. As MXFP4, with an FP8-aware scale codec; FP4 nibbles dominate the residual.
:::
::

::u-page-section
---
class: dark:bg-gradient-to-b from-neutral-950 to-neutral-900
---
:::UPageCTA
---
title: Get started
description: Run `pip install ptwm`. Requires Python 3.12+, PyTorch 2.6+, Linux.
links:
  - label: Quickstart
    to: /quickstart
    trailingIcon: i-lucide-arrow-right
  - label: View on GitHub
    to: https://github.com/khwstolle/ptwm
    target: _blank
    variant: subtle
    icon: i-simple-icons-github
class: dark:bg-neutral-950
---
:::
::
