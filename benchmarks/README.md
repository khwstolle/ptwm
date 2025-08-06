# PTWM benchmark harness

Manual performance + validation harnesses, refactored from earlier
`scratch_*.py` profiling scripts. These are **not** part of the pytest suite:
they read real model shards and are tuned via environment variables. For the
synthetic, assertion-based benchmarks that *do* run in CI, see
`tests/ptwm/benchmarks/`.

Run any benchmark as a module from the repo root:

```bash
python -m benchmarks.<name> [args]
```

## Environment

| Variable | Meaning |
| --- | --- |
| `RAYON_NUM_THREADS` | PTWM's Rust thread count (read once at pool init). Set to `1` for single-thread numbers, to the core count for scaling. Several scripts are meant to be run twice, at `1` and at `ncores`, to read the scaling curve. |
| `PTWM_PROFILE=1` | Emit per-pass encode timings from the Rust side to stderr (read by `encode_pass_profile`). |
| `PTWM_HF_CACHE` | Root of the HuggingFace shard cache. Defaults to the standard hub cache location (`~/.cache/huggingface/hub`). |
| `PTWM_BENCH_OUT` | Output dir. Read by both `validate_realmodel` (default `/tmp/ptwm_bench_out`) and `encode_fast` (default `/tmp/ptwm_fast_bench_out`), each with its own default. |
| `RSS_DIR` | Pre-compressed `.ptwm` directory for `probe_rss dec`. |

## Benchmarks

### Encode

| Module | Needs a shard? | What it measures |
| --- | --- | --- |
| `encode_stages` | yes (BF16 shard) | Stage-by-stage encode breakdown (framework / transform / codec / trial-encode cost) + cProfile. Run with `RAYON_NUM_THREADS=1`. |
| `encode_throughput` | yes (BF16 shard) | Single-tensor encode MB/s + ratio across PTWM configurations, at matched algorithms. `RAYON_NUM_THREADS=1`. |
| `encode_threading` | yes (BF16 shard) | PTWM at matched thread counts. Run twice (`RAYON_NUM_THREADS=1` and `=ncores`). |
| `encode_chunks` | yes (BF16 shard) | `streaming_chunk` size sweep: speed/ratio tradeoff of chunk-parallel encode. Run twice for scaling. |
| `encode_model` | yes (multi-tensor shard) | Full-model aggregate throughput: PTWM one `compress_model` call, inter-tensor parallel. |
| `encode_pass_profile` | yes (multi-tensor shard) | Per-pass Rust timings for one full-model encode. Run with `PTWM_PROFILE=1 RAYON_NUM_THREADS=32`. |
| `encode_fast` | yes (real shards) | `fast=True` forced-rANS+chunked vs production full-menu encode; verifies the fast path stays lossless. |

The single-tensor encode scripts take a `.safetensors` path and use the first
BF16 tensor ≥ 50 MB; the full-model scripts use BF16 tensors up to a 3 GB cap.

### Decode

| Module | Needs a shard? | What it measures |
| --- | --- | --- |
| `decode_throughput` | yes (multi-tensor shard) | Full-model decode MB/s: PTWM `decode_model` (parallel). Verifies a lossless roundtrip before timing. Run twice for scaling. |

### Validation

| Module | Needs a shard? | What it measures |
| --- | --- | --- |
| `validate_dtype` | no (synthetic) | Losslessness + ratio across BF16/FP16/FP32/FP8 through the production streaming path. |
| `validate_scale {small,large,edge}` | `large` only | `small`: many tiny tensors; `large`: a 72B-class shard (scaling + RSS); `edge`: single-tensor files, non-float dtypes, degenerate shapes. Every mode verifies a bit-exact roundtrip. |
| `validate_realmodel` | yes (real shards) | Real-model validation matrix: ratio / encode / decode / losslessness across Mistral-7B (BF16), gpt-oss-20b (MXFP4), Nemotron (FP8), Qwen2.5-72B (BF16). |
| `probe_rss {enc,dec}` | yes (Qwen shard) | Clean peak-RSS probe, one operation per process. |

`validate_dtype` and `validate_scale small`/`edge` need no shards and run
anywhere the extension is built; use them as a local smoke test of the harness.
