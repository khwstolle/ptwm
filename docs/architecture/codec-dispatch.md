# Codec dispatch

After the PPG chain runs, each terminal plane has a role and a byte
buffer. The codec dispatcher decides which entropy coder to apply.
This document explains the dispatch policy, why trial-encode is
preferred over a learned predictor, and how the per-plane state
flows through the wire format.

## The dispatcher contract

The dispatcher takes a plane (bytes plus descriptor) and a list of
candidate codecs, runs each candidate's encode method, and keeps
the smallest output. The byte count is the only ranking signal.
This is the trial-encode approach. It looks wasteful at first but
the alternatives have problems that the simple approach avoids.

The natural alternative is a predictor that scores codecs from
descriptor features without actually running them. A predictor lets
the dispatcher pick a codec in microseconds rather than running N
full encodes. The catch is that a predictor that is right almost
all the time produces compression regressions on the cases where
it is wrong, and a regression in the dispatcher means worse
compression for every user whose tensor shape falls outside the
distribution the predictor was tuned for. Trial-encode is right by
construction: the chosen codec is the smallest, by definition.

The CPU cost is bounded by the number of candidate codecs, which
is small. On a typical compress call the dispatcher trials four or
five codecs per plane. The `should_attempt` hook on the codec
trait gives each codec a chance to skip its own encode on inputs
where a cheap statistical estimate predicts no gain; this trims
the constant factor without changing the correctness guarantee
that trial-encode provides.

## Codec menu construction

The menu of candidates per plane is built by `menu_for_descriptor`
in `compressor.rs`. The function looks at the plane's role and
element width and returns the codecs that accept that descriptor
shape. The acceptance predicate is the `accepts` method on each
`PlaneCodec` trait implementation.

The role-based filtering matters because some codecs are heavily
specialised. The `Order1ScaleAC` codec only accepts scale planes in
FP8 formats (`E4M3` or `E5M2`), where the mantissa bits give a
continuous distribution with per-row autocorrelation the coder can
exploit. It rejects `E8M0` scales (pure 8-bit exponents used by
MXFP4 block scales) because that structure is outside its design
domain, and it rejects every non-scale role outright. The
`PerGroupCodebook` codec only accepts packed FP4 planes. The
`accepts` predicate keeps these codecs out of the trial set for
planes they cannot handle.

The shared-state codecs (`Order1ScaleAC`, `PerGroupCodebook`) have
two trial flavours: the inline variant where each plane carries its
own state bytes, and the shared variant where multiple planes share
state stored in the file prelude. Pass 2 of the compressor
(`fit_shared`) builds candidate shared states across all tensors,
and Pass 3 re-trials each plane against both flavours. The shared
variant wins when the per-tensor state cost exceeds the file-level
state cost amortised across N tensors.

## The `forced_codec` and `allow_codec_ids` knobs

`CompressorOptions` exposes two ways to override the dispatcher's
choice. They serve different purposes and they do not compose.

`forced_codec` bypasses dispatch entirely. The compressor runs the
forced codec on every plane regardless of role. This is what the
`--method huffman` and `--method zstd` CLI flags wire up. The use
case is benchmarks: a paper that wants to report "what does PTWM do
with only Huffman?" sets `forced_codec = Some(Huffman)` and the
result is a clean single-codec baseline. The forced codec must
accept any descriptor; in practice only Identity and Huffman are
safe choices for arbitrary inputs.

`allow_codec_ids` constrains the menu without bypassing dispatch.
The compressor still runs trial-encode but skips codecs whose id is
not in the allow list. An empty intersection on a plane is a hard
error rather than a fallback to Identity, because falling back
silently would falsify the ablation measurement the option exists
to support. This is the path that `ablation/_bench.py` uses for
variant policies.

The two knobs cannot both be set. The compiler does not enforce
this because the constructor surface is permissive; the rule is
documented in `CompressorOptions` and the compressor ignores
`allow_codec_ids` when `forced_codec` is set.

## State sources

Each plane record carries a `state_source` field with four values:

`None` means the codec needs no state. Identity, Huffman, rANS, and
Zstd all fall here. The codec's encode and decode methods see only
the plane bytes; there is no per-plane configuration. This is the
common case and the fast path.

`Inline` means the codec's state is stored alongside the plane in
`inline_state_bytes`. Order1ScaleAC and PerGroupCodebook both
produce a small probability table or codebook on encode, store it
inline, and read it back on decode. The state bytes are bounded
(typically a few hundred bytes) so storing them per plane is cheap
when only a few tensors use the codec.

`Shared` means the state lives in the file prelude, identified by
`state_info` as the prelude entry index. The plane carries no state
bytes itself. This is the right choice when many planes share the
same state distribution; the cost is paid once per file.

`External` means the state lives outside the container, retrieved
by the host application. This is used by the integration adapters
when state can be reconstructed from the host model's metadata
(for example, NVFP4 global scales that the safetensors header
already carries). External state is not currently exercised by
PTWM's own writers.

## Why `state_format_version`

Every state-using codec has a version byte stamped into the plane
record. The version is independent of the codec id, since the same
codec can change its state format without changing the canonical
id. This decoupling matters because the codec id is wire-stable and
cannot change, but the state format may evolve when a better
encoding is discovered.

The version byte is checked at decode time; a state version the
codec does not recognise is rejected with a clear error rather than
attempting a decode that would silently produce wrong bytes. The
state version is the only versioning surface in the codec contract;
deliberate.

## The router and the dispatcher

There are two dispatch surfaces in the codebase, and the distinction
matters when reading the compressor code:

The compressor's dispatcher (`compressor.rs::trial_encode_plane`)
picks among in-tree built-in codecs at encode time. It builds the
menu from the descriptor, trials each candidate, and writes the
plane record with the winner's `codec_id`.

The container reader's router (`flavor/router.rs::PlaneCodecRouter`)
looks up a canonical id at decode time and returns a
`DispatchedPlaneCodec` that the decode loop can invoke. The router
handles both built-in canonical ids and third-party canonical ids;
the latter loads a WASM or native bundle.

The two paths converge at the `PlaneCodec` trait: the router's
`BuiltinAdapter` wraps the same trait implementations the
compressor's dispatcher uses, so the decode behaviour for built-in
codecs is byte-for-byte identical to what the encoder produced. For
third-party codecs the router uses a `ThirdPartyPlaneCodec` adapter
that bridges the byte-buffer flavor ABI to the in-tree trait.
