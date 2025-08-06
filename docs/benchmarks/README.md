# Benchmark results

PTWM compression numbers on representative real-world checkpoints.

## Setup

* **Corpus:**
  * `openai/gpt-oss-20b`: MXFP4-quantized MoE, 13.76 GB raw
    (FP4 nibbles + E8M0 scales + BF16 layers).
  * `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4`: NVFP4-quantized
    Llama derivative, 31.07 GB raw (FP4 nibbles + FP8-E4M3 scales).
* **Configurations evaluated:**
  * `huffman_only`: PTWM with the preprocessing graph forced
    to passthrough; pure byte-Huffman per tensor. Isolates the
    contribution of the preprocessing graph.
  * `microscale_full`: PTWM with the production
    preprocessing graph and the MICROSCALE method enabled.
* **Hardware:** CPU-only host, 16 cores, 64 GB RAM.

## Compression ratios

Ratio is compressed bytes ÷ raw bytes. Lower is better.

| Model | huffman_only | microscale_full |
|---|---|---|
| `openai/gpt-oss-20b`                                | 0.8804 | **0.8603** |
| `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4`    | 0.9089 | **0.9003** |

**Combined-corpus ratio (44.83 GB raw):** PTWM
`microscale_full` lands at 0.8881.

## Encode time

Single-process encode time on the same hardware:

| Configuration | gpt-oss-20b | Nemotron-NVFP4 |
|---|---|---|
| `huffman_only`     | 519.33 s | 1296.62 s |
| `microscale_full`  | 281.65 s |  677.23 s |

`microscale_full` runs at ≈ 50 MB/s in this configuration, suitable
for cold-storage and distribution but not for online recompression.

## What the data says

1. **Microscaling formats are already near-incompressible at the
   byte level.** Trained FP4 / NVFP4 nibble-packed bytes are close to
   uniform; no amount of plane-aware coding extracts much further.
2. **The remaining lossless headroom is in the scale plane.** The
   MICROSCALE method's scale-plane codec dispatch compresses E8M0
   scales to ≈ 0.16 of original on real-world models. (Order-1
   ScaleAC itself is gated to FP8-format scales, namely E4M3 / E5M2
   on NVFP4, and does not apply to E8M0, which is a pure 8-bit
   exponent with no exploitable per-row autocorrelation; the E8M0
   ratio below comes from the generic per-plane codec trial-encode.)
   The whole-model effect is bounded by the scale-mass fraction
   (≈ 5%), which is why the on-disk gain over byte-Huffman is small
   in absolute terms.
3. **The PTWM preprocessing graph adds value over pure
   byte-Huffman, but only modestly on microscaling formats.**
   `huffman_only` (graph forced to passthrough) hits 0.8804 on
   gpt-oss-20b vs 0.8603 with the full graph, a 2.0 percentage-point
   delta, most of which is the scale codec; the rest is dtype-aware
   nibble routing.

## On-disk vs. algorithmic size

The numbers above report on-disk `.ptwm` directory size, including
the multi-tensor container header, tensor index, per-plane codec
dispatch records, and payload hashes. Container overhead runs roughly
1.6 percentage points on gpt-oss-20b, the price of features the raw
entropy coders do not provide (random access, hash verification,
multi-tensor packaging).

For a pure entropy-coder comparison, subtract the container overhead:
`microscale_full` on gpt-oss-20b drops from 0.8603 to roughly 0.844
algorithmically. The on-disk number is reported here because it
matches what users measure with `du`.

## Pipeline validation: throughput, scaling, memory, dtype coverage

Validation of the parallelized encode/decode pipeline across dtypes, scale,
and edge cases. All numbers measured on a single 32-core CPU node;
`encode`/`decode` are full-model throughput through the safetensors file
integration (`compress_safetensors_file` + `read_sharded_ptwm`). Raw payload
sizes; lossless verified by bit-exact roundtrip against the source
safetensors.

Companion data: [`validation-realmodel.csv`](validation-realmodel.csv).

### Real-model headline (production path, 32 threads)

| Model | dtype | Size | Ratio | Encode | Decode | Lossless |
|---|---|---:|---:|---:|---:|:--:|
| Mistral-7B-v0.1 | bf16 | 4.5 GB | 0.6609 | 45 MB/s | 1111 MB/s | ✓ |
| gpt-oss-20b | mxfp4 | 4.8 GB | 0.8932 | 77 MB/s | 1635 MB/s | ✓ |
| Nemotron-Super-120B | fp8-e4m3 | 5.0 GB | 0.7165 | 26 MB/s | 627 MB/s | ✓ |
| Qwen2.5-72B-Instruct | bf16 | 3.8 GB | 0.6604 | 24 MB/s | 793 MB/s | ✓ |

Decode (the model-load hot path) runs at **0.6–1.6 GB/s**: every tensor in a
shard is decoded across the rayon pool in one call.

### Fast encode mode

`compress_safetensors_file(fast=True)` forces rANS as the only per-plane codec
(skipping the Huffman/Zstd/Identity trial menu) and chunks output. The
transform-chain search still runs, so the ratio is preserved on float models.

| Model | Encode (prod → fast) | Speedup | Ratio (prod → fast) |
|---|---:|---:|---:|
| Mistral-7B bf16 | 36.7 → 148.1 MB/s | 4.0× | 0.6609 → 0.6608 |
| gpt-oss-20b mxfp4 | 55.7 → 471.5 MB/s | 8.5× | 0.8932 → 0.8986 |

Pure-float models keep parity ratio; quant scale planes stay lossless but lose
the `Order1ScaleAC` edge (+0.6% on MXFP4). Fast mode is tuned for normal-sized
models; on many tiny tensors (e.g. 2000 × 64×64) forced rANS loses to Huffman
on small exponent planes (0.709 vs 0.678), so the default trial menu is better
there.

### Scaling (Qwen2.5-72B shard, 3.8 GB, 32 tensors)

| | 1 thread | 32 threads | Scaling |
|---|---:|---:|---:|
| Encode (prod) | 3.7 MB/s | 25–29 MB/s | ~7.9× |
| Encode (fast) | 15.1 MB/s | 85 MB/s | ~5.7× |
| Decode | 169 MB/s | 656–793 MB/s | ~3.9× |

Encode scales near-linearly per-tensor; the sub-linear tail is the small,
uneven tensor count (32 tensors, the largest bounds the critical path). Decode
scales ~4×; previously it was flat (~100–130 MB/s at any thread count)
because tensors were decoded one at a time in Python.

### Peak memory (and how to bound it)

Clean per-operation peak RSS for the 3.8 GB shard, as a multiple of shard size:

| Operation | 1 thread | 32 threads |
|---|---:|---:|
| Decode | 2.0× (7.8 GB) | 3.3× (12.6 GB) |
| Encode (fast) | — | 5.7× (21.7 GB) |
| Encode (production) | 5.3× (20.2 GB) | 10.9× (41.6 GB) |

**Decode is cheap.** Production encode at high thread count is the memory peak,
because Pass-2 shared-state fitting needs every transformed plane resident
before any record can be written (it cannot stream), and parallel trial-encode
buffers add to that as thread count rises.

Two zero-cost dials bound encode peak when memory is tight on a large shard:

- **`fast=True`** roughly halves it (5.7× vs 10.9×) by forcing one codec.
- **Lower `RAYON_NUM_THREADS`** trades encode speed for memory (5.3× at 1
  thread vs 10.9× at 32).

Shards are capped ~4–5 GB and processed one at a time, so the per-shard peak is
what matters, not the whole model.

### Dtype & edge-case coverage (all lossless)

- **Floats:** bf16, fp16, fp32, fp8-e4m3, fp8-e5m2, MXFP4, NVFP4.
- **Non-float:** int8, int16, int32, int64, uint8.
- **Shapes:** scalar-ish `(1,)`, odd `(3,7,11)`, `(1,1,1,1)`, single-tensor
  files, and **empty 0-dimension tensors** `(0,)` / `(0,16)` (the last fixed a
  `torch.frombuffer` zero-length-buffer bug).

## Per-feature CSVs

Smaller, per-feature benchmark CSVs live alongside this file:

* `preprocessing-ablations.csv`: preprocessing-graph ablations across dtypes.
* `plane-entropy.csv`: per-plane entropy measurements (raw vs
  shuffle vs reorder+split).
* `method-comparison.csv`: synthetic-data comparison across
  PTWM methods (`IDENTITY`, `HUFFMAN`, `RANS`, `ZSTD`).
* `codec-comparison.csv`: per-codec ratio and throughput on
  real models.
* `random-access-tradeoff.csv`: per-tensor vs monolithic compression
  trade-off.
* `validation-realmodel.csv`: companion data for the pipeline validation
  section above.
