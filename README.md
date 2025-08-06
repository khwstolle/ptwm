# PTWM

[![PyPI](https://img.shields.io/pypi/v/ptwm.svg)](https://pypi.org/project/ptwm/)
[![Python](https://img.shields.io/pypi/pyversions/ptwm.svg)](https://pypi.org/project/ptwm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/khwstolle/ptwm/actions/workflows/python-tests.yml/badge.svg)](https://github.com/khwstolle/ptwm/actions/workflows/python-tests.yml)

A lossless compression library for PyTorch model weights. PTWM
exploits the byte-level structure of IEEE 754 floats and the layout of
microscaling formats (MXFP4 / NVFP4) to recover compression that
generic codecs leave on the table, with fast parallel encode and
decode through a native Rust extension.

The output format, `.ptwm`, is a multi-tensor container with
random-access lookup, hash-verified payloads, and per-tensor codec
dispatch.

Benchmarks live under [`docs/benchmarks/`](docs/benchmarks/); design
docs live under [`docs/architecture/`](docs/architecture/).

## What this library does and does not do

| Workload | Expected ratio | Recommendation |
|---|---|---|
| `bfloat16` / `float16` / `float32` weights | ≈ 0.55–0.75 | **Use PTWM.** Exponent-plane separation is the right tool. |
| `float8_e4m3fn` / `float8_e5m2` weights | ≈ 0.70–0.85 | **Use PTWM.** Nibble-split + plane-aware Huffman. |
| MXFP4 weights (FP4 nibbles + E8M0 scales) | ≈ 0.86–0.90 | The main win is the random-access container, not the raw ratio: the nibble plane is already near the byte-level entropy floor. |
| NVFP4 weights (FP4 nibbles + FP8-E4M3 scales) | ≈ 0.90 | Same caveat as MXFP4: the FP4 nibble plane is near-uniform and dominates the size. |
| Already-compressed data (PNG, ZIP, encrypted) | ≈ 1.0 | No codec helps. |

Microscaling formats already sit near the entropy floor at the byte
level, so no general-purpose lossless codec extracts more than a few
percent from them. See [`docs/benchmarks/README.md`](docs/benchmarks/README.md)
for the benchmark numbers.

## How it works

Neural network weights have a hidden structure: trained exponents
cluster around a narrow range while mantissa bits are near-random.
PTWM exposes this structure through a typed preprocessing
graph (PPG) before entropy coding.

**1. Bit reorder.** IEEE 754 floats place the sign bit between the
exponent and mantissa, straddling a byte boundary. The bit reorder
shifts the exponent to the most-significant position with the sign
tucked below it.

For `float32` and `bfloat16` (8-bit exponent), the exponent occupies
a full byte after reorder:

```
IEEE 754 fp32:  [S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM]
After reorder:  [EEEEEEEE S MMMMMMMMMMMMMMMMMMMMMMM]
```

For `float8_e4m3fn` and `float8_e5m2` (4–5 bit exponent), the reorder
moves the exponent into the high nibble of each byte:

```
FP8-E4M3FN:    [S EEEE MMM]
After reorder: [EEEE SMMM ]   ← exponent in high nibble
```

**2. Byte / nibble split.** The split deinterleaves the reordered data
into separate planes. Each plane is entropy-coded independently, using
whichever of Huffman, rANS, or Zstd compresses smaller, except the
NVFP4 scale plane, which gets a dedicated arithmetic coder tuned to
its scale distribution (see the dtype table below).

- `float32`: 4-plane round-robin byte split (plane 3 = exponent)
- `float16` / `bfloat16`: 2-plane byte split (plane 1 = exponent)
- `float8_e4m3fn` / `float8_e5m2`: 2-plane nibble split (plane 0 = exponent nibbles)
- `float4_e2m1fn_x2` (MXFP4 / NVFP4): nibble-pair split for the weight plane; the paired E8M0 (MXFP4) or FP8-E4M3 (NVFP4) scale tensor is a separate plane

The exponent plane typically holds only ~20 unique values out of 256
(~2.6 bits/byte entropy on real-world trained weights), while mantissa
planes look near-random. Nearly all the compression savings on float
weights come from the exponent plane. Microscaling formats flip the
picture: nibble bytes are near-uniform, and almost all the savings
come from the scale plane.

The two-step decomposition is a standard byte-shuffle preprocessing
step, plus a bit-reorder step that cleanly separates the exponent from
the sign. Without it, the sign contaminates the exponent byte.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI](#cli)
- [Python API](#python-api)
- [Integrations](#integrations)
- [Configuration](#configuration)
- [Supported dtypes](#supported-dtypes)
- [Extensions](#extensions)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Benchmarks](#benchmarks)

## Installation

```sh
pip install ptwm
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.6. If you need a specific
PyTorch build (CPU-only, a particular CUDA version), install it from
the appropriate [PyTorch index](https://pytorch.org/get-started/locally/)
before or alongside `ptwm`.

## Quickstart

```python
import torch
from ptwm import Compressor, Decompressor, CompressionConfig, Format

config = CompressionConfig(input_format=Format.TORCH)
compressor = Compressor(config)
decompressor = Decompressor()

# float32 / bfloat16 / float16
tensor = torch.randn(1024, 1024, dtype=torch.bfloat16)
restored = decompressor.decompress(compressor.compress(tensor))
assert torch.equal(tensor, restored)

# float8: automatic nibble split (exponent / mantissa planes)
fp8 = torch.randn(1024, 1024).to(torch.float8_e4m3fn)
restored_fp8 = decompressor.decompress(compressor.compress(fp8))
assert torch.equal(fp8.view(torch.uint8), restored_fp8.view(torch.uint8))

# float4 (MXFP4 / NVFP4 packed weights, PyTorch ≥ 2.6).
# When paired with their FP8 / E8M0 block scales (via the same model),
# the .ptwm container routes each through the appropriate codec.
if hasattr(torch, "float4_e2m1fn_x2"):
    fp4 = torch.randint(0, 256, (512, 512), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    restored_fp4 = decompressor.decompress(compressor.compress(fp4))
    assert torch.equal(fp4.view(torch.uint8), restored_fp4.view(torch.uint8))
```

For multi-tensor models, use the safetensors integration below; it
emits a single `.ptwm` container with random-access tensor lookup.

## CLI

The `ptwm` command-line tool compresses and decompresses files.
Compressed output uses the `.ptwm` suffix.

**Compress a `.safetensors` model (single multi-tensor `.ptwm` file):**

```sh
ptwm compress model.safetensors
# produces model.safetensors.ptwm
```

**Compress any other file:**

```sh
ptwm compress model.bin
# produces model.bin.ptwm
```

**Compress every `.safetensors` file in a directory:**

```sh
ptwm compress /path/to/model/
```

**Decompress:**

```sh
ptwm decompress model.safetensors.ptwm
# restores model.safetensors
```

**Delta compression** (compress relative to a reference file):

```sh
ptwm compress model_v2.safetensors --delta model_v1.safetensors
```

**Tune chains for a specific model** (extends the production chain set
with explorer-discovered candidates; discovered chains are stored inline
in the output and decode on any PTWM install):

```sh
ptwm compress model.safetensors --explore --out model.ptwm/
ptwm chains cache info        # inspect cached discoveries
ptwm chains export bundle     # share a tuned chain set
ptwm chains import bundle     # adopt a peer's chain set
```

Run `ptwm compress --help` or `ptwm decompress --help` for the
full option list.

## Python API

```python
from ptwm import (
    Compressor, Decompressor,
    CompressionConfig, DecompressionConfig,
    Method, Format,
)

# Compress raw bytes (dtype hint required for byte-reordering)
config = CompressionConfig(
    method=Method.AUTO,           # AUTO, HUFFMAN, RANS, ZSTD, IDENTITY, or MICROSCALE
    input_format=Format.BYTE,     # BYTE, TORCH, NUMPY, or FILE
    bytearray_dtype="bfloat16",   # dtype hint for byte-reordering (BYTE format only)
    threads=8,                    # parallelism (default: up to 16 cores)
)
compressor = Compressor(config)
compressed: bytes = compressor.compress(data)

# Decompress: the return type (bytes / torch.Tensor / np.ndarray) is
# recovered automatically from metadata embedded at compress time.
decompressor = Decompressor(DecompressionConfig(threads=8))
restored: bytes = decompressor.decompress(compressed)
```

`AUTO` (the default) runs the full per-plane trial-encode: every codec
candidate for the tensor's dtype and plane roles competes, and the
smallest result wins. `HUFFMAN`, `RANS`, `ZSTD`, and `IDENTITY` force
that one codec instead of trial-encoding. `MICROSCALE` routes MXFP4 /
NVFP4 paired tensors through the dedicated scale-plane codec (the
weight-nibble plane still trial-encodes).

### Random access

```python
from ptwm.random_access import TensorIndex

idx = TensorIndex.open("model.safetensors.ptwm")
weight = idx.get_tensor("layer_5.weight")
for name, tensor in idx.stream_tensors(["embed.weight", "lm_head.weight"]):
    ...
```

The container's tensor index is small (a manifest of name → offset
records), so opening a bundle and fetching one tensor touches only
that tensor's bytes.

## Integrations

Everything below lives in `ptwm.integrations`, imported explicitly.
Nothing here is re-exported from the top-level `ptwm` package.

### Safetensors

`SafeOpen` is a drop-in replacement for `safetensors.safe_open`: same
interface (`keys()`, `get_tensor()`, context manager), transparently
reading `.ptwm` containers, `.safetensors` shells wrapping a `.ptwm`
blob, and legacy per-tensor-compressed `.safetensors` files.

```python
from ptwm.integrations import SafeOpen

with SafeOpen("model.safetensors", framework="pt") as f:
    tensor = f.get_tensor("weight")
```

### HuggingFace Transformers

Decompress a cached snapshot once, then load with plain `transformers`.
No patching is active during the actual model load:

```python
from huggingface_hub import snapshot_download
from ptwm.integrations import materialize_hf_cache

snapshot_dir = snapshot_download("org/model-name")
materialize_hf_cache(snapshot_dir)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(snapshot_dir)
```

### Monkey-patching (when you don't control the call site)

If third-party code calls `safetensors.torch.safe_open` or
`transformers.modeling_utils.load_state_dict` internally, so you
can't swap in `SafeOpen` or run `materialize_hf_cache` first, both
integrations have an opt-in patch that installs transparent `.ptwm`
support into the target library for the lifetime of the process:

```python
from ptwm.integrations import patch_safetensors
patch_safetensors()
```

```python
from ptwm.integrations import patch_transformers
patch_transformers()
```

## Configuration

A few `CompressionConfig` options beyond the ones shown above:

| Option | Default | Description |
|---|---|---|
| `compression_threshold` | `0.95` | Skip compression and store raw bytes if the achieved ratio exceeds this (not worth the container overhead) |
| `codec_menu` | `None` (full menu) | Restrict per-plane trial-encode to a specific `CodecId` list, e.g. for ablations |
| `is_streaming` / `streaming_chunk` | `False` / `1048576` (1 MiB) | Encode a large tensor as independently-decodable chunks instead of one pass |

`CompressionConfig` also has delta-compression and lossy-compression
fields; see its docstring for the full list.

## Supported dtypes

| PyTorch dtype | Strategy | Notes |
|---|---|---|
| `float32` | Bit reorder + 4-plane byte split | Exponent → plane 3 |
| `bfloat16` / `float16` | Bit reorder + 2-plane byte split | Exponent → plane 1 |
| `float8_e4m3fn` / `float8_e5m2` | Bit reorder + 2-plane nibble split | Exponent → high nibble |
| `float4_e2m1fn_x2` (weight nibbles) | Passthrough, per-plane trial-encode | `PerGroupCodebook` competes against Huffman / rANS on the nibble-packed byte stream |
| MXFP4 scale (`uint8`, E8M0) | Passthrough, per-plane trial-encode | Pure 8-bit exponent with no row structure to exploit; compresses to ≈ 0.16 of original via the generic codec menu |
| NVFP4 scale (FP8-E4M3) | `Order1ScaleAC` (per-row lag-1 arithmetic coding) | Mantissa bits give exploitable row autocorrelation that E8M0 lacks |
| `int8`, `uint8`, `int16`, `int32`, `int64`, `bool` | Single stream | No byte split or bit reorder yet; whatever wins per-plane trial-encode on the raw bytes |

MXFP4 / NVFP4 tensors are compressed as a pair: the packed FP4 weight
nibbles (`float4_e2m1fn_x2`) and their block-scale tensor (E8M0 for
MXFP4, FP8-E4M3 for NVFP4) are two separate planes with the codec
choices above.

## Extensions

PTWM has a native multi-language extension system. New codecs,
preprocessing transforms, classifiers, scoring functions, and more can
be added as separate signed bundles in **WASM**, **native cdylib**, or
**Python host** form, without modifying PTWM core.

- **Design + per-kind ABI:** [`docs/architecture/extension-system.md`](docs/architecture/extension-system.md)
- **Reference contributions:** [`extensions/`](extensions/)

### Researchers

```sh
pip install ptwm[ext-dev]
ptwm ext init my-codec --lang rust --kind plane_codec
cd my-codec
ptwm ext build
ptwm ext sign --key ~/.ptwm/keys/mykey.priv
ptwm ext pack
```

### Operators

```sh
ptwm trust add --bundled                   # one-time consent to the bundled keyring
ptwm ext install ./my-codec-0.1.0.tar.zst  # or https://, oci://, git+SHA, or pip name
ptwm ext list                              # confirm
```

### Ablation

```sh
ptwm bench compress model.bin \
    --baseline policies/baseline.toml \
    --variant policies/no-huffman.toml \
    --trials 3 --out bench-out/
ptwm bench attribute model.bin --leave-one-out --out attr-out/
```

## Repository layout

- **`crates/ptwm-core/`**: Rust library for the preprocessing graph,
  entropy coders, and container format.
- **`crates/ptwm-py/`**: PyO3 bindings that build the `ptwm._core`
  native extension module.
- **`python/ptwm/`**: the Python package, covering the API, CLI, and
  integrations (HuggingFace `transformers`, `safetensors`).
- **`extensions/`**: reference implementations for the extension
  system, one per contribution kind.
- **`docs/architecture/`**: design docs covering the wire format,
  preprocessing graph, codec dispatch, extension system, and trust
  model.
- **`docs/benchmarks/`**: benchmark results and per-feature CSVs.
- **`tests/`**: the pytest suite.
- **`benchmarks/`**: manual performance and validation harnesses.
- **`fuzz/`**: `cargo-fuzz` targets for the container format and
  entropy coders.

## Development

The development environment lives in `pyproject.toml` + `uv.lock`,
driven by [uv](https://docs.astral.sh/uv/).

**Prerequisites:**

- Python ≥ 3.12 (build supports 3.12 and 3.13)
- A stable Rust toolchain (`rustc`, `cargo`) for building the
  `ptwm._core` native extension
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install the prerequisites via your system package manager or
[rustup](https://rustup.rs/). uv can provision a matching Python
(`uv python install 3.13`) if your system lacks one. Then:

```sh
uv sync --dev
source .venv/bin/activate
```

**Pre-commit hooks.** Prek manages the hook set:

```sh
uv tool install prek
prek install
prek run --all-files
```

## Benchmarks

[`docs/benchmarks/README.md`](docs/benchmarks/README.md) summarises the
benchmark numbers on representative real-world models, and the same
directory holds per-feature CSVs (ablations, plane entropy, codec
comparison across the library's own methods, random-access overhead).

## License

MIT. See [`LICENSE`](LICENSE).
