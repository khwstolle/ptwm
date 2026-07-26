# The Preprocessing Graph (PPG)

The PPG is the most important architectural idea in PTWM. Almost all
of the compression-ratio story happens before the entropy coder runs.
This document explains why PTWM uses a typed graph of preprocessing
ops rather than a fixed pipeline, and why the chain table looks the
way it does.

## The problem the PPG solves

A naive approach to lossless weight compression feeds the raw bytes
of a tensor into Zstd and stops. On trained BF16 weights this gets
about a 1.05x ratio, because the raw byte stream looks near-random.
The randomness is real at the byte level: IEEE 754 places the sign
bit between the exponent and the mantissa, straddling a byte
boundary. A high-order byte of a BF16 weight carries one sign bit,
the top seven bits of the exponent, and nothing else useful. The
distribution across the tensor looks roughly uniform.

Reordering the bits so the exponent ends up in its own byte gives an
exponent byte plane with very low entropy on trained weights. Most
trained weights cluster around magnitudes 2^-7 to 2^-3, so the
exponent bytes take only a handful of distinct values. Huffman
coding on that plane reaches about 2.6 bits per byte, which is what
most of PTWM's compression ratio comes from.

This generalises. Some tensors compress better with the mantissa
further split into separate planes; some compress better when the
scale bytes are treated separately from the value bytes; FP8 needs a
nibble split rather than a byte split because the exponent is four
bits, not eight. The PPG is the data structure that records which
of these transformations was applied to which tensor, so the decoder
can reverse them.

## Why a graph and not a sequence

The simplest model for preprocessing is a sequence of
transformations applied in order, like a Unix pipeline. This model
is appealing for its conceptual clarity but it cannot express the
preprocessing PTWM actually needs.

The first reason a graph is needed is fan-out. A byte split takes
one input plane and produces N output planes. A nibble split
produces two output planes. These outputs flow through different
downstream paths. The mantissa byte typically goes straight to a
terminal, while the exponent byte may go through a delta predictor
before reaching its terminal. A sequence cannot express this; a
typed graph can.

The second reason is cross-tensor edges. Delta compression
references a second tensor as the predictor. The dependency is on a
different tensor's bytes, so the graph node needs to declare which
of its inputs is the local tensor and which is the reference. A
sequence has no notion of multiple inputs.

The third reason is descriptor propagation. Each plane carries a
`PlaneDescriptor` with role, element width, and layout. The
descriptors flow through the graph and the validator confirms at
chain-construction time that every op's outputs are typed in a way
its downstream consumers can accept. This catches malformed chains
without running them, which would otherwise mean a wasted forward
pass on the whole tensor.

## Op catalogue

The op enum (`OpId` in `transforms/op.rs`) is sealed at seventeen
values. Adding an eighteenth is a wire-format change. The current
catalogue covers:

The control ops `Source` and `Terminal` mark chain boundaries. The
source carries the input dtype and tensor shape. Each terminal
carries the role bytes that the codec dispatcher uses to score
candidate codecs.

The bit reorder family (`BitReorderIeee16`, `BitReorderIeee32`,
`BitReorderFp8E4M3`, `BitReorderFp8E5M2`) does the per-element bit
shuffle that moves the exponent into its own byte (or nibble). The
specialisation by dtype lets each op know the exact bit layout
without an extra parameter.

The split family (`ByteSplit`, `NibbleSplit`) factors a plane into
per-significance planes. These are intended to run after a bit
reorder, since the inputs are most useful when each output
corresponds to a single significance level.

The delta family is split between cross-tensor ops (`XorDelta`,
`FloatDelta`) and the intra-plane op (`IntDelta`). The intra-plane
one consumes a single input; the cross-tensor ones consume two
(the local plane and a reference plane).

The passthrough ops (`BytePassthrough`, `MxFp4Deinterleave`,
`Reshape`, `Concat`, `IndexBitwidthPack`) handle reshape, dtype-
specific reinterpretation, and packed-integer cases. `EntropyEstimate`
is a measurement no-op that computes per-byte Shannon entropy of its
input and stores it in `vendor_bytes`; it is available to any chain
but is not currently wired into a production chain or the explorer
(see the chain cache's own entropy bucketing below, which computes
the same quantity directly on raw tensor bytes rather than through
this op).

## The chain table

`PRODUCTION_CHAINS` in `python/ptwm/preprocessing/_chains.py` is the
hand-curated table mapping `(dtype, classifier_role)` to a list of
candidate `Chain`s. The dispatcher runs every candidate chain
through trial-encode and keeps the smallest output per tensor. A
list rather than a single chain is necessary because the best chain
depends on tensor statistics that the table cannot predict.

For example, BF16 STANDARD currently has three candidates: the plain
bit-reorder-plus-byte-split chain, the same chain with `IntDelta`
applied to the exponent terminal, and the same chain with
`PredictorXor` applied to the exponent terminal. Some weight rows
compress harder with one variant; some with another. Trial-encode
picks the winner per tensor with no risk of regression, since the
dispatcher only commits to a candidate that beats the baseline byte
count.

The `(dtype, role)` keying exists because different tensors within
the same model need different chains. The scale tensors in a
microscaling model want the `PerGroupCodebook` path, not byte
splitting. The classifier (`python/ptwm/classify/`) decides the
role; the chain table dispatches on it.

## Chain explorer

The explorer in `python/ptwm/preprocessing/_explorer.py` is a BFS
over chain mutations, used to extend the hand-curated table with
new candidates that a specific model benefits from. The explorer is
gated behind `--explore` because it is expensive: each candidate
chain requires a full forward pass plus trial encode.

Discovered chains persist to `$XDG_CACHE_HOME/ptwm/chains/` keyed by
`(dtype, role, signature_bucket)`. The signature bucket is computed
by `signature_bucket_for()` (`python/ptwm/preprocessing/_cache.py`)
from the Shannon entropy of the tensor's raw bytes, quantized into
0.5-bit-wide buckets. The keying assumption is that two tensors with
similar entropy fall in the same bucket and benefit from the same
chain, so an explorer run on one model extends the candidate set for
future runs on models with similar weight distributions. This
assumption is a design choice, not yet validated against measured
hit rates across model families.

The explorer is also the path by which hand-curated chains get into
the table. `ptwm chains promote` prints the structural breakdown of
a discovered chain so a maintainer can hand-write a named builder.
The intent is that the curated table absorbs whatever the explorer
keeps re-discovering.

## Wire format of a chain

`Chain.to_bytes()` serialises a chain to the format
`compress_model` accepts. The format carries:

```
[ nodes_count u8                 ]
[ edges_count u8                 ]
[ terminals_count u8             ]
[ nodes_buf_len u16              ]
[ edges_buf_len u16              ]
[ terminals_buf_len u16          ]
[ for each node: op_id u16 + params_len u8 + params bytes ]
[ for each edge: 4 indices + role_override flag + vendor_bytes ]
[ for each terminal: node_idx + output_idx + role bytes ]
```

The format is intentionally compact. A typical chain serialises to
under one hundred bytes. The serialised chains live inline in the
container's extension table, so chain size shows up as fixed
overhead on small files.

The `chain_blob_from_legacy` bridge in the Rust side handles
conversion from the historical opcode-only encoding to the current
extension-table-prefixed encoding. The bridge exists because PTWM
shipped the chain bytes before the extension table did; backward
compatibility requires accepting both forms on read.

## What the validator does

`chain/validate.rs` enforces a small set of structural rules at
chain-construction time, before the chain ever runs:

Every node has either an explicit source declaration or at least one
incoming edge. There are no orphan nodes.

Every cross-tensor op (`XorDelta`, `FloatDelta`) declares a
dependency index that is in range for the tensor record's
`dependencies` field. Out-of-range dep indices are rejected at
write time, since they would produce undecodable files.

Descriptor types flow consistently. The validator runs the
`propagate_descriptors` method of each op against the upstream
descriptors and checks the downstream consumers' expectations. A
type mismatch is a chain-construction error, not a runtime panic.

Terminals declare their role bytes explicitly. The role bytes flow
to the dispatcher to pick a codec; an undeclared role would force
the dispatcher to use a generic fallback that defeats most of the
chain's purpose.

These rules are weak by typical static-typing standards; the
graph's full correctness requires runtime checks too. The trade-off
is that the validator runs in microseconds and catches the common
authoring mistakes early. Stronger static checks would require a
typed graph IR that PTWM has not invested in.
