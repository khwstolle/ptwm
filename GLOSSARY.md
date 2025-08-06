# Ubiquitous Language

## Preprocessing (PPG)

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **PPG (Preprocessing Graph)** | The typed DAG concept that routes every tensor through preprocessing ops before entropy coding | preprocessing pipeline, transform graph |
| **Chain** | A concrete, wire-serializable instance of a PPG — a typed DAG whose nodes are preprocessing ops | pipeline, graph (when meaning a specific instance) |
| **ChainBuilder** | A factory that constructs a `Chain` for a given dtype/classifier combination | builder |
| **PRODUCTION_CHAINS** | The static table mapping `(dtype_code, ClassifierRole)` to a list of `ChainBuilder`s used in normal operation | chain table, default chains |
| **Explorer (`explore_chains` / `ExploreOptions`)** | The BFS generative search that discovers new candidate chains beyond `PRODUCTION_CHAINS` | chain search, BFS explorer |
| **Chain cache** | User-local persistence of explorer-discovered chains under `$XDG_CACHE_HOME/ptwm/chains/`, so later compresses skip re-running BFS | discovery cache |
| **Preprocessing op** | A single node/primitive in a `Chain`: bit reorder, byte split, nibble split, or byte passthrough | transform, stage |
| **Bit reorder** | Preprocessing op that moves the IEEE 754 sign bit off the exponent/mantissa byte boundary into a dedicated byte or nibble | bit shuffle |
| **Byte split** | Preprocessing op that separates same-significance bytes of a tensor's elements into distinct planes | byte deinterleave |
| **Nibble split** | Like byte split, but at 4-bit granularity — used for FP8 exponents | nibble deinterleave |
| **Byte passthrough** | Preprocessing op that forwards bytes unmodified into a plane, with no bit/byte manipulation | passthrough, identity op |
| **Terminal** | A `Chain`'s exit node; carries the `PlaneRole` that must be re-tagged onto the resulting plane's descriptor | leaf node, chain output |

## Planes, codecs, and entropy coding

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Plane** | A byte stream produced by a `Chain`, tagged with a `PlaneRole`, that is entropy-coded independently | stream, byte plane |
| **PlaneRole** | The semantic tag on a plane (e.g. exponent, mantissa, sign, raw, scale) that governs which codecs are legal on it | plane type, plane kind |
| **PlaneCodec** | The trait/contract every entropy coder implements to compress/decompress a single plane | codec (only when the plane-level contract is meant, not the dispatch-level `Method`) |
| **Dispatcher** | The component that trial-encodes each plane against every registered `PlaneCodec` (Identity, Huffman, rANS, Zstd, …) and keeps the smallest payload | selector, codec picker |
| **Entropy coder** | One of the two pure-Rust peer coders — Huffman or rANS — that share the 4-stream interleaved framing and the `PlaneCodec` contract | compressor (avoid — ambiguous with the container writer) |
| **Method (`EnumMethod`)** | The wire-format hint selecting a compression strategy at the tensor/container level: `ZSTD` (forces the Zstd plane codec as fallback) or `MICROSCALE` (routes MXFP4/NVFP4 scale-paired tensors through the scale-plane codec) | mode, strategy — do not conflate with a per-plane **codec** of the same name (e.g. the `ZSTD` codec) |
| **Codec menu** | The opt-in registry of codecs a dispatcher may consider, which can include codecs absent from the default `ALL` set via `accepts()` | codec set |
| **Order1ScaleAC** | A `PlaneCodec` specialized for order-1 arithmetic coding of microscale (E8M0) scale planes | scale codec |
| **PerGroupCodebook** | A `PlaneCodec` that builds a separate codebook per group of elements | group codebook |
| **Stream frame** | The shared 4 × u32 jump-table framing (`stream_frame.rs`) that both Huffman and rANS use to interleave their four backward bitstreams | 4-stream framing |
| **CompressOutcome** | The result type an entropy coder returns after compressing a plane | outcome |

## Container and format

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **`.ptwm` container** | The multi-tensor on-disk format: shared-state prelude, tensor index, and per-plane codec dispatch | ptwm file, container format |
| **Header** | The wire-format structure describing a `.ptwm` container's layout | wire header |
| **Format** | The input/output data kind a tensor is read from or written to: `BYTE`, `TORCH`, `NUMPY`, or `FILE` | data format (only when meaning this specific enum) |
| **Mode `a` / Mode `b`** | For safetensors input: `a` writes a native `.ptwm` directory; `b` writes a `.safetensors` shell wrapping a `.ptwm` blob | native mode / shell mode |
| **TensorIndex** | The `random_access/` structure enabling per-tensor random access into a container | tensor lookup |
| **Delta compression** | Reference-frame (XOR/delta residual) compression of a tensor against a prior version | reference-frame compression |

## Integration and tooling

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **`patch_safetensors()`** | Patches `safetensors.torch.safe_open` with a `SafeOpen` wrapper that transparently decompresses marked tensors | safetensors patch |
| **`patch_transformers()`** | Patches `transformers.modeling_utils.load_state_dict` for HuggingFace model loading | transformers patch |
| **MultiProcessPatcher** | Shared utility both integration patches use to apply monkey-patches safely across processes | patcher |
| **`ptwm chains promote`** | CLI command that prints a structural breakdown of a discovered chain so a maintainer can hand-write it into `PRODUCTION_CHAINS` | promote command |
| **`ptwm chains export` / `import`** | CLI commands that move tuned chain sets between users; `import` validates every chain through `is_legal_chain` | chain export/import |

## Relationships

- A **PPG** is instantiated as a **Chain**; a **Chain** is built by a **ChainBuilder** drawn from **PRODUCTION_CHAINS** or discovered by the **Explorer**.
- A **Chain** consists of one or more **preprocessing ops**, ending in one or more **Terminals**, each of which produces a **Plane** tagged with a **PlaneRole**.
- The **Dispatcher** trial-encodes each **Plane** against every candidate **PlaneCodec** and keeps the smallest result, recorded as a **CompressOutcome**.
- Multiple **Planes** (and their **CompressOutcomes**) are packed into a **`.ptwm` container**, described by its **Header** and indexed by a **TensorIndex**.
- A **Method** (`ZSTD`, `MICROSCALE`) constrains or forces which **PlaneCodec** the **Dispatcher** may pick, independent of the general per-plane trial-encoding logic.

## Example dialogue

> **Dev:** "Why is the exponent plane compressing so much better than the mantissa plane?"
>
> **Domain expert:** "Because the **Chain** for that dtype starts with a **bit reorder** op — it pulls the sign bit off the exponent/mantissa byte boundary. Once that's done, the exponent **Plane** carries almost all the entropy, so whichever **PlaneCodec** the **Dispatcher** picks — Huffman or rANS — gets most of the savings there."
>
> **Dev:** "So the mantissa plane doesn't need a smart codec?"
>
> **Domain expert:** "Right, it usually ends up on Identity or Zstd. The **Dispatcher** doesn't know that in advance — it trial-encodes every **Plane** against every candidate **PlaneCodec** and keeps the smallest **CompressOutcome**."
>
> **Dev:** "And MICROSCALE — is that a codec too?"
>
> **Domain expert:** "No, that's a **Method**, not a **PlaneCodec**. It's a wire-format hint that routes MXFP4/NVFP4 scale-paired tensors to the `Order1ScaleAC` codec specifically. Don't confuse it with the `ZSTD` **codec** — `ZSTD` is also a **Method** name, but it means 'force the Zstd plane codec as fallback' at the tensor level, which is a different layer than the per-plane codec selection."

## Flagged ambiguities

- **"ZSTD"** is overloaded: it names both a `Method` (`EnumMethod::ZSTD`, forcing the Zstd codec as a tensor-level fallback) and a `PlaneCodec` (the actual Zstd implementation registered with the dispatcher). Recommendation: when precision matters, say "the `ZSTD` method" vs. "the Zstd plane codec."
- **"Chain" vs. "PPG"** are sometimes used interchangeably in the codebase description. Recommendation: use **PPG** for the general concept/architecture, and **Chain** only for a concrete instantiated DAG (a specific value, e.g. one produced by a `ChainBuilder` or the Explorer).
- **"Codec"** is used at two layers: the per-plane `PlaneCodec` trait implementation (Identity, Huffman, rANS, Zstd, Order1ScaleAC, PerGroupCodebook) and, informally, as shorthand for a `Method`. Recommendation: reserve bare "codec" for the `PlaneCodec` layer, and always say "method" for the tensor-level `EnumMethod` selection.
- **"Compressor"** could mean the entropy coder (Huffman/rANS) or the container writer (`compressor.rs`, `compress_model`). Recommendation: say "entropy coder" for Huffman/rANS, and "container writer" or "`compress_model`" for the multi-tensor `.ptwm` writer.
