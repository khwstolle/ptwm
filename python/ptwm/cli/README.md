# PTWM CLI

PTWM provides a unified command-line interface for compressing and
decompressing PyTorch model weights.

## Installation

The CLI ships as the `ptwm` console script when the `ptwm` package is
installed:

```bash
pip install ptwm
ptwm --help
```

## Available commands

### Compression

`ptwm compress` routes based on the input path:

- **Directory**: compresses all `.safetensors` files in the directory.
- **`.safetensors` file**: tensor-by-tensor compression into a `.ptwm`
  container.
- **Other file**: byte-stream compression, output `<input>.ptwm`.
- **`--delta <reference>`**: compresses the difference between the input
  and a reference file.

#### Examples

```bash
# Compress a single raw file
ptwm compress model.bin

# Compress a safetensors model (tensor-by-tensor)
ptwm compress model.safetensors

# Compress every safetensors file in a directory
ptwm compress ./my_model_dir

# Delta compression
ptwm compress model.bin --delta base_model.bin

# Hugging Face cache compression
ptwm compress safetensors --model ibm-granite/granite-7b-instruct --hf_cache
```

### Decompression

`ptwm decompress` restores `.ptwm` blobs back to their original form:

```bash
# Decompress a single file
ptwm decompress model.bin.ptwm

# Decompress a tensor-by-tensor safetensors container
ptwm decompress model.ptwm.safetensors

# Decompress every .ptwm file in a directory
ptwm decompress ./my_model_dir

# Delta decompression
ptwm decompress model_delta_base.ptwm --delta base_model.bin
```

## Advanced options

Both `compress` and `decompress` support these flags:

- `--dtype`: input data type for raw-byte streams (`bfloat16`,
  `float16`, `float32`, `float8_e4m3fn`, `float8_e5m2`).
- `--method`: compression method: `HUFFMAN` (default), `RANS`,
  `IDENTITY`, `ZSTD`, `AUTO`, `MICROSCALE`. All methods run through the
  native `ptwm._core` extension; `ZSTD` forces the in-tree Zstd plane
  codec as a generic fallback.
- `--threads`: worker thread count.
- `--force`: overwrite existing output files.
- `--delete`: remove the original file after a successful operation.
- `--verification`: verify the compressed output decompresses to the
  exact input.
- `--hf_cache`: preserve Hugging Face cache symlink layout.
