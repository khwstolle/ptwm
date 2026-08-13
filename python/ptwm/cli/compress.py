import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ptwm.cli.utils import (
    RED,
    RESET,
    check_and_install_ptwm,
    parse_streaming_chunk_size,
    replace_in_file,
)
from ptwm.codecs import CodecId

_CODEC_NAME_MAP: dict[str, CodecId] = {
    "identity": CodecId.Identity,
    "huffman": CodecId.Huffman,
    "rans": CodecId.Rans,
    "zstd": CodecId.Zstd,
    "per-group-codebook": CodecId.PerGroupCodebook,
    "order1-scale-ac": CodecId.Order1ScaleAC,
}


def _parse_codec_menu(arg: str | None) -> list[CodecId] | None:
    """Parse a comma-separated list of codec names into a list of CodecId.

    Returns None for a None argument (full menu); raises ValueError on
    unknown names.
    """
    if arg is None:
        return None
    out: list[CodecId] = []
    for token in arg.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in _CODEC_NAME_MAP:
            raise ValueError(f"unknown codec name: {token!r}")
        out.append(_CODEC_NAME_MAP[key])
    return out


def _explore_options_from_args(args):  # noqa: ANN001
    """Build an ``ExploreOptions`` from CLI args, or ``None`` if --explore is off."""
    if not getattr(args, "explore", False):
        return None
    from ptwm.preprocessing import ExploreOptions  # noqa: PLC0415

    return ExploreOptions(
        max_candidates_per_dtype=args.explore_max_candidates,
        time_budget_ms=args.explore_budget_ms,
        allow_cross_tensor=args.explore_allow_cross_tensor,
    )


def _format_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _print_explore_summary(
    audit,  # noqa: ANN001
    input_bytes: int,
    output_bytes: int,
) -> None:
    """Print a post-compression summary when --explore was used.

    Reported:
    - Explorer discovery count per dtype + role group.
    - Final compressed size vs. input size (real numbers from disk).

    Not reported:
    - Whether the discovered chain or the production chain won per tensor —
      the dispatcher does not surface the winning index.
    """
    discovered = list(audit.discovered_chains())
    print()  # noqa: T201
    if not discovered:
        print(  # noqa: T201
            "Explored 0 additional chain candidates beyond the production "
            "defaults. The default table covered every (dtype, role) seen."
        )
    else:
        total_tried = sum(d.n_candidates_tried for d in discovered)
        print(  # noqa: T201
            f"Explored {total_tried} additional chain candidates across "
            f"{len(discovered)} (dtype, role) group(s) beyond the production "
            f"defaults:"
        )
        for d in discovered:
            print(  # noqa: T201
                f"  - dtype=0x{d.dtype_code:04X} role={d.role}: "
                f"{d.n_candidates_tried} candidate(s), first seen on "
                f"{d.sample_tensor_name}"
            )

    if input_bytes > 0:
        ratio = output_bytes / input_bytes
        print(  # noqa: T201
            f"Output: {_format_bytes(output_bytes)} "
            f"(input: {_format_bytes(input_bytes)}, ratio {ratio:.4f})"
        )
    else:
        print(f"Output: {_format_bytes(output_bytes)}")  # noqa: T201

    if discovered:
        print(  # noqa: T201
            "Discovered chains are stored inline in the .ptwm container and "
            "decode on any PTWM install."
        )


def compress_file(
    input_file: str,
    dtype: str = "bfloat16",
    streaming_chunk_size: int | str = 1048576,
    delete: bool = False,
    force: bool = False,
    hf_cache: bool = False,
    method: str = "HUFFMAN",
    verification: bool = False,
    test: bool = False,
    is_streaming: bool = False,
    threads: int | None = None,
    quiet: bool = False,
    codec_menu: list[CodecId] | None = None,
    codec: str | None = None,
    device: int | None = None,
) -> None:
    """Compress a single file."""
    from ptwm import (  # noqa: PLC0415
        CompressionConfig,
        Compressor,
        DecompressionConfig,
        Decompressor,
        Method,
    )

    chunk_size = parse_streaming_chunk_size(streaming_chunk_size)
    full_path = Path(input_file)
    if not full_path.exists():
        if not quiet:
            print(f"{RED}File {input_file} not found.{RESET}", file=sys.stderr)  # noqa: T201
        raise FileNotFoundError(input_file)

    compressed_path = full_path.with_suffix(full_path.suffix + ".ptwm")
    if not test and not force and compressed_path.exists():
        user_input = (
            input(f"{compressed_path} already exists; overwrite (y/n)? ")
            .strip()
            .lower()
        )
        if user_input not in ("yes", "y"):
            return

    output_file = compressed_path
    compressor = Compressor(
        CompressionConfig(
            bytearray_dtype=dtype,
            is_streaming=is_streaming,
            streaming_chunk=chunk_size,
            method=Method(method),
            threads=threads,
            codec_menu=codec_menu,
            codec=codec,
            device=device,
        )
    )

    file_size_before = 0
    file_size_after = 0

    if not test:
        with full_path.open("rb") as infile, output_file.open("wb") as outfile:
            chunk = infile.read()
            file_size_before += len(chunk)
            compressed_chunk = compressor.compress(chunk)
            if compressed_chunk:
                file_size_after += len(compressed_chunk)
                outfile.write(compressed_chunk)
    else:
        test_buffer = bytearray()
        with full_path.open("rb") as infile:
            chunk = infile.read()
            file_size_before += len(chunk)
            compressed_chunk = compressor.compress(chunk)
            if compressed_chunk:
                file_size_after += len(compressed_chunk)
                test_buffer += compressed_chunk

    if verification:
        # `device` is the compression-side ordinal already passed to
        # CompressionConfig above; this round-trip check always decodes on
        # CPU, so it is left unset on DecompressionConfig rather than
        # passed through.
        decompressor = Decompressor(DecompressionConfig(threads=threads))
        if test:
            with full_path.open("rb") as f:
                file_data2 = f.read()
            if decompressor.decompress(bytes(test_buffer)) != file_data2:
                msg = "Decompressed file should be equal to original file."
                raise RuntimeError(msg)
        else:
            with full_path.open("rb") as infile, output_file.open("rb") as outfile:
                file_data1 = infile.read()
                file_data2 = outfile.read()
            decompressed_data = decompressor.decompress(file_data2)
            if file_data1 != decompressed_data:
                msg = "Decompressed file should be equal to original file."
                raise RuntimeError(msg)

    if delete and not hf_cache:
        full_path.unlink()

    if hf_cache:
        try:
            snapshot_path = full_path.parent
            blob_name = snapshot_path / full_path.readlink()
            output_file.rename(blob_name)
            output_file.symlink_to(blob_name)
            if full_path.exists():
                full_path.unlink()
        except OSError as err:
            msg = f"Error reorganizing Hugging Face cache: {err}"
            raise RuntimeError(msg) from err


def compress_file_delta(
    input_file: str,
    delta_file: str,
    dtype: str = "bfloat16",
    streaming_chunk_size: int | str = 1048576,
    delete: bool = False,
    force: bool = False,
    hf_cache: bool = False,
    method: str = "HUFFMAN",
    verification: bool = False,
    test: bool = False,
    is_streaming: bool = False,
    threads: int | None = None,
    codec_menu: list[CodecId] | None = None,
    codec: str | None = None,
    device: int | None = None,
) -> None:
    """Compress a file using delta compression."""
    from ptwm import (  # noqa: PLC0415
        CompressionConfig,
        Compressor,
        DecompressionConfig,
        Decompressor,
        Method,
    )

    chunk_size = parse_streaming_chunk_size(streaming_chunk_size)
    full_path = Path(input_file)
    delta_path = Path(delta_file)
    missing = [str(p) for p in (full_path, delta_path) if not p.exists()]
    if missing:
        msg = f"missing input file(s): {', '.join(missing)}"
        print(f"{RED}{msg}{RESET}", file=sys.stderr)  # noqa: T201
        raise FileNotFoundError(msg)
    if delete and not hf_cache:
        msg = f"{RED}Delete not supported yet for delta compression.{RESET}"
        raise ValueError(msg)

    folder_path = full_path.parent
    input_filename = full_path.name
    delta_filename = delta_path.name
    output_file = folder_path / (
        input_filename[:-4] + "_delta_" + delta_filename + ".ptwm"
    )

    if not test and not force and output_file.exists():
        user_input = (
            input(f"{output_file} already exists; overwrite (y/n)? ").strip().lower()
        )
        if user_input not in ("yes", "y"):
            return

    compressor = Compressor(
        CompressionConfig(
            bytearray_dtype="float32" if dtype else "bfloat16",
            is_streaming=is_streaming,
            streaming_chunk=chunk_size,
            delta_compressed_type="file",
            method=Method(method),
            threads=threads,
            codec_menu=codec_menu,
            codec=codec,
            device=device,
        )
    )

    with full_path.open("rb") as f:
        file_data = f.read()
    compressed_data = compressor.compress(file_data, delta_second_data=delta_file)

    if verification:
        # `device` is not forwarded here either; see the note in
        # `compress_file`.
        decompressor = Decompressor(
            DecompressionConfig(
                delta_second_data=delta_path.read_bytes(),
                threads=threads,
            )
        )
        with full_path.open("rb") as f:
            file_data2 = f.read()
        decompressed_data = decompressor.decompress(compressed_data)
        if decompressed_data != file_data2:
            msg = "Decompressed file should be equal to original file."
            raise RuntimeError(msg)

    if not test:
        with output_file.open("wb") as f_out:
            f_out.write(compressed_data)

    if hf_cache:
        try:
            snapshot_path = full_path.parent
            blob_name = snapshot_path / full_path.readlink()
            output_file.rename(blob_name)
            output_file.symlink_to(blob_name)
            if full_path.exists():
                full_path.unlink()
        except OSError as err:
            msg = f"Error reorganizing Hugging Face cache: {err}"
            raise RuntimeError(msg) from err


def compress_safetensors_file(
    filename: str,
    delete: bool = False,
    force: bool = False,
    hf_cache: bool = False,
    method: str | None = None,
    threads: int | None = None,
    quiet: bool = False,
    codec_menu: list[CodecId] | None = None,
    codec: str | None = None,
    device: int | None = None,
) -> None:
    """Compress a safetensors file."""
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    from ptwm import (  # noqa: PLC0415
        CompressionConfig,
        Compressor,
        Method,
    )
    from ptwm._config import Format  # noqa: PLC0415
    from ptwm.utils._safetensors import (  # noqa: PLC0415
        COMPRESSED_DTYPE,
        COMPRESSION_METHOD,
        build_compressed_tensor_info,
        set_compressed_tensors_metadata,
    )

    full_path = Path(filename)
    if not filename.endswith(".safetensors"):
        if not quiet:
            print(  # noqa: T201
                f"{RED}File {filename} is not a .safetensors file.{RESET}",
                file=sys.stderr,
            )
        return

    compressed_path = full_path.with_name(
        full_path.name[: -(len(".safetensors"))] + ".ptwm.safetensors"
    )
    if not force and compressed_path.exists():
        user_input = (
            input(f"{compressed_path} already exists; overwrite (y/n)? ")
            .strip()
            .lower()
        )
        if user_input not in ("yes", "y"):
            return

    tensors = {}
    compressed_tensor_infos = {}
    compression_method = (
        Method(method) if method is not None else Method(COMPRESSION_METHOD)
    )

    compressor_cache = {}

    with safe_open(filename, "pt", "cpu") as f:
        for name in f.keys():  # noqa: SIM118
            tensor = f.get_tensor(name)

            compressed_tensor_info = build_compressed_tensor_info(tensor)

            dtype_str = str(tensor.dtype).replace("torch.", "")
            if dtype_str not in compressor_cache:
                compressor_cache[dtype_str] = Compressor(
                    CompressionConfig(
                        input_format=Format.TORCH,
                        bytearray_dtype=dtype_str,
                        method=compression_method,
                        threads=threads,
                        codec_menu=codec_menu,
                        codec=codec,
                        device=device,
                    )
                )
            compressor = compressor_cache[dtype_str]

            uncompressed_size = tensor.element_size() * tensor.nelement()
            compressed_buf = compressor.compress(tensor)
            compressed_size = len(compressed_buf)

            if compressed_size >= uncompressed_size:
                tensors[name] = tensor
                continue

            compressed_tensor = torch.frombuffer(compressed_buf, dtype=COMPRESSED_DTYPE)
            tensors[name] = compressed_tensor
            compressed_tensor_infos[name] = compressed_tensor_info

        metadata = f.metadata()

    set_compressed_tensors_metadata(compressed_tensor_infos, metadata)
    save_file(tensors, str(compressed_path), metadata)

    if delete and not hf_cache:
        full_path.unlink()

    if hf_cache:
        try:
            snapshot_path = full_path.parent
            blob_name = snapshot_path / full_path.readlink()
            compressed_path.rename(blob_name)
            compressed_path.symlink_to(blob_name)
            if full_path.exists():
                full_path.unlink()
        except OSError as err:
            msg = f"Error reorganizing Hugging Face cache: {err}"
            raise RuntimeError(msg) from err


def compress_path(
    suffix: str,
    dtype: str = "bfloat16",
    streaming_chunk_size: int | str = 1048576,
    path: str = ".",
    delete: bool = False,
    recursive: bool = False,
    force: bool = False,
    max_processes: int = 1,
    hf_cache: bool = False,
    model: str = "",
    branch: str = "main",
    method: str = "HUFFMAN",
    verification: bool = False,
    test: bool = False,
    is_streaming: bool = False,
    threads: int | None = None,
    file_compression: bool = False,
    codec_menu: list[CodecId] | None = None,
    codec: str | None = None,
    device: int | None = None,
) -> None:
    """Compress all files with the given suffix in the specified path."""
    overwrite_first = True
    file_list = []
    chunk_size = parse_streaming_chunk_size(streaming_chunk_size)
    search_path = Path(path)

    if model:
        if not hf_cache:
            msg = "Must specify --hf_cache when using --model"
            raise ValueError(msg)
        try:
            from huggingface_hub import scan_cache_dir  # noqa: PLC0415
        except ImportError as err:
            msg = "huggingface_hub not found. Please pip install huggingface_hub."
            raise ImportError(msg) from err
        cache = scan_cache_dir()
        repo = next((repo for repo in cache.repos if repo.repo_id == model), None)

        if repo is not None:
            repo_path = Path(repo.repo_path)
            try:
                with (repo_path / "refs" / branch).open("r") as ref:
                    rev_hash = ref.read().strip()
            except FileNotFoundError:
                msg = f"Branch {branch} not found in repo {model}"
                raise FileNotFoundError(msg) from None
            search_path = repo_path / "snapshots" / rev_hash

    if recursive:
        files_to_check = search_path.rglob(f"*{suffix}")
    else:
        files_to_check = search_path.glob(f"*{suffix}")

    for full_path in files_to_check:
        file_name = full_path.name
        if file_compression:
            compressed_name = file_name + ".ptwm"
        else:
            compressed_name = file_name[: -(len(".safetensors"))] + ".ptwm.safetensors"
        compressed_path = full_path.parent / compressed_name

        if not test and not force and compressed_path.exists():
            if overwrite_first:
                overwrite_first = False
                user_input = (
                    input("Compressed files already exists; overwrite them all (y/n)? ")
                    .strip()
                    .lower()
                )
                if user_input in ("y", "yes"):
                    force = True
            if not force and not test:
                user_input = (
                    input(f"{compressed_path} already exists; overwrite (y/n)? ")
                    .strip()
                    .lower()
                )
                if user_input not in ("y", "yes"):
                    continue
        file_list.append(str(full_path))

    if file_list and hf_cache:
        try:
            from transformers.utils import (  # noqa: PLC0415
                SAFE_WEIGHTS_INDEX_NAME,
                WEIGHTS_INDEX_NAME,
            )
        except ImportError as err:
            msg = "Transformers not found. Please pip install transformers."
            raise ImportError(msg) from err

        new_replace = f"{suffix}.ptwm" if file_compression else f"ptwm.{suffix}"
        for index_name in (SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME):
            index_path = search_path / index_name
            if index_path.exists():
                blob_path = search_path / index_path.readlink()
                replace_in_file(file_path=blob_path, old=f"{suffix}", new=new_replace)

    if file_compression:
        compression_func = compress_file
        comp_args = (
            dtype,
            chunk_size,
            delete,
            True,
            hf_cache,
            method,
            verification,
            test,
            is_streaming,
            threads,
            True,  # quiet
            codec_menu,
            codec,
            device,
        )
    else:
        compression_func = compress_safetensors_file
        comp_args = (
            delete,
            True,
            hf_cache,
            method,
            threads,
            True,
            codec_menu,
            codec,
            device,
        )

    failures: list[tuple[str, BaseException]] = []
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        future_to_file = {
            executor.submit(compression_func, file, *comp_args): file
            for file in file_list[:max_processes]
        }
        remaining_files = deque(file_list[max_processes:])
        while future_to_file:
            for future in as_completed(future_to_file):
                file = future_to_file.pop(future)
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((file, exc))
                    print(
                        f"{RED}error compressing {file}: {exc!r}{RESET}",
                        file=sys.stderr,
                    )
                if remaining_files:
                    next_file = remaining_files.popleft()
                    future_to_file[
                        executor.submit(compression_func, next_file, *comp_args)
                    ] = next_file
    if failures:
        print(
            f"{RED}{len(failures)} of {len(file_list)} file(s) failed to compress{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_size(s: str | int) -> int:
    """Parse '5GB', '500MB', '1024' (bytes) into an int."""
    if isinstance(s, int):
        return s
    s = s.strip().upper()
    multipliers = {"TB": 1024**4, "GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for suffix, mul in multipliers.items():
        if s.endswith(suffix):
            num_str = s[: -len(suffix)].strip()
            return int(float(num_str) * mul)
    return int(s)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: parse flags and dispatch to the appropriate codec.

    Distinct from ``handle_compress`` (driven by the ``ptwm compress`` sub-
    command parser). Owns a dedicated argparse parser exposing ``--mode``,
    ``--classify-*``, ``--max-shard-size``, ``--out`` alongside a single
    positional input path.

    For ``.safetensors`` inputs, delegates to
    ``ptwm.integrations.compress_safetensors_file``; all other input types
    fall back to ``compress_file``.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="ptwm compress",
        description="Compress a model checkpoint.",
    )
    parser.add_argument("input", help="Input file path (.safetensors or raw bytes).")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (required for .safetensors inputs).",
    )
    parser.add_argument(
        "--mode",
        choices=["a", "b"],
        default="a",
        help="Output mode: 'a' = native .ptwm shards (default), 'b' = .safetensors shell.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        dest="max_shard_size",
        help="Maximum uncompressed byte budget per shard (default: 5GB).",
    )
    parser.add_argument(
        "--quant-config",
        type=str,
        default=None,
        dest="quant_config",
        help="Path to hf_quant_config.json.",
    )
    parser.add_argument(
        "--use-quant-config",
        action="store_true",
        dest="use_quant_config",
        help="Auto-discover hf_quant_config.json adjacent to the input file.",
    )
    parser.add_argument(
        "--weights-config",
        type=str,
        default=None,
        dest="weights_config",
        help="Path to a weights ptwm config file.",
    )
    parser.add_argument(
        "--classify-rule",
        type=str,
        action="append",
        default=[],
        dest="classify_rule",
        metavar="GLOB=ROLE",
        help="Explicit classification rule '<glob>=<role>' (repeatable).",
    )
    parser.add_argument(
        "--classify-heuristic",
        action="store_true",
        dest="classify_heuristic",
        help="Enable the built-in name-pattern heuristic classifier.",
    )

    args = parser.parse_args(argv)

    from ptwm.classify import (  # noqa: PLC0415
        ClassifierChain,
        ExplicitFlagsClassifier,
        HeuristicClassifier,
        HfQuantConfigClassifier,
        PtwmConfigClassifier,
    )

    steps = []
    quant_path = Path(args.quant_config) if args.quant_config else None
    if args.use_quant_config and quant_path is None:
        quant_path = Path(args.input).parent / "hf_quant_config.json"
    if quant_path is not None and quant_path.exists():
        steps.append(HfQuantConfigClassifier.from_path(quant_path))
    if args.weights_config:
        steps.append(PtwmConfigClassifier.from_path(args.weights_config))
    if args.classify_rule:
        steps.append(ExplicitFlagsClassifier.from_flag_strings(args.classify_rule))
    if args.classify_heuristic:
        steps.append(HeuristicClassifier())
    chain = ClassifierChain(steps)

    input_path = Path(args.input)
    if input_path.suffix == ".safetensors":
        if args.out is None:
            print(  # noqa: T201
                f"{RED}Error: --out is required for .safetensors inputs.{RESET}",
                file=sys.stderr,
            )
            return 1
        from ptwm.integrations import compress_safetensors_file  # noqa: PLC0415

        compress_safetensors_file(
            input_path,
            args.out,
            mode=args.mode,
            classifier=chain,
            max_shard_size=_parse_size(args.max_shard_size),
        )
        return 0

    # Fallback: raw-bytes compression via the existing helper.
    compress_file(input_file=str(input_path))
    return 0


def add_compress_parser(subparsers):
    parser = subparsers.add_parser("compress", help="Compress a file or directory")
    parser.add_argument("path", type=str, help="Path to file or directory")
    parser.add_argument("--delta", type=str, help="Path to delta file")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2"],
        default="bfloat16",
        help="Specify the data type. Default is bfloat16.",
    )
    parser.add_argument(
        "--streaming_chunk_size",
        type=str,
        help="Optional streaming chunk size (e.g., 1MB). Default is 1MB",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete single file instead of compression",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting when compressing.",
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="Indicate if the file is in the Hugging Face cache.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["HUFFMAN", "RANS", "IDENTITY", "ZSTD", "AUTO", "MICROSCALE"],
        default="HUFFMAN",
        help="Specify the method to use. Default is HUFFMAN.",
    )
    parser.add_argument(
        "--verification",
        action="store_true",
        help="Verify that compression can be decompressed correctly.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Do not write the compression to a file.",
    )
    parser.add_argument(
        "--is_streaming",
        action="store_true",
        help="Compress using streaming.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="The amount of threads to be used.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursive search on all subdirectories",
    )
    parser.add_argument(
        "--max_processes",
        type=int,
        default=1,
        help="The amount of maximum processes.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--model_branch",
        type=str,
        default="main",
        help="Specify the model branch. Default is 'main'",
    )
    parser.add_argument(
        "--file_compression",
        action="store_true",
        help="Compress the file as a whole, not per tensor.",
    )
    parser.add_argument(
        "--codec-menu",
        type=str,
        default=None,
        help=(
            "Comma-separated codec names to restrict the trial-encode menu. "
            "Valid: identity, huffman, rans, zstd, per-group-codebook, "
            "order1-scale-ac. Ignored when --method selects a forced-codec path."
        ),
    )
    parser.add_argument(
        "--codec",
        default=None,
        help="Select one codec explicitly, by name (e.g. zstd) or canonical id. "
        "Skips the automatic per-plane codec search.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device ordinal for GPU-resident codecs. Defaults to 0. "
        "Must match the input tensor's device when that tensor is already on GPU.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help=(
            "Run the chain explorer for each (dtype, role) pair and try the "
            "discovered candidates alongside the production defaults. "
            "Picks the smallest of all candidates per tensor. Only applies to "
            ".safetensors inputs; writes a multi-tensor .ptwm directory. "
            "Discovered chains are stored inline in the output and decode on "
            "any PTWM install."
        ),
    )
    parser.add_argument(
        "--explore-budget-ms",
        type=int,
        default=5000,
        dest="explore_budget_ms",
        help="Per (dtype, role) time budget for chain exploration in ms (default: 5000).",
    )
    parser.add_argument(
        "--explore-max-candidates",
        type=int,
        default=32,
        dest="explore_max_candidates",
        help="Maximum explorer-discovered candidates per (dtype, role) (default: 32).",
    )
    parser.add_argument(
        "--explore-allow-cross-tensor",
        action="store_true",
        dest="explore_allow_cross_tensor",
        help="Allow the explorer to consider cross-tensor (delta) ops.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        dest="no_cache",
        help=(
            "Bypass the user-local chain cache for this run. Discovered "
            "chains from --explore are NOT written back, and previously "
            "cached chains are NOT used as candidates."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output path for .safetensors inputs. With --mode a, the output "
            "is a directory (default <input>.ptwm/). With --mode b, the "
            "output is a single .safetensors-shaped file."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["a", "b", "legacy"],
        default=None,
        dest="ptwm_mode",
        help=(
            "Container mode for .safetensors inputs: "
            "'a' = native .ptwm directory (default for --explore), "
            "'b' = .ptwm-blob wrapped in a .safetensors shell, "
            "'legacy' = legacy per-tensor wrapper (deprecated). "
            "When unset and --explore is off, defaults to 'legacy' for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["ptwm", "lmdb", "webdataset"],
        default="ptwm",
        dest="store_format",
        help=(
            "Storage format for .safetensors inputs: 'ptwm' (default, native "
            "container), 'lmdb' (memory-mapped KV store), or 'webdataset' "
            "(tar-shard archives for streaming)."
        ),
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["blob", "exploded"],
        default="blob",
        dest="lmdb_layout",
        help="LMDB value layout (--format lmdb): 'blob' (default) or 'exploded'.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default=None,
        dest="store_max_shard_size",
        help="WebDataset max shard size (e.g. 2GB). --format webdataset only.",
    )
    parser.add_argument(
        "--lmdb-map-size",
        type=str,
        default=None,
        dest="lmdb_map_size",
        help="LMDB environment map size (e.g. 8GB). --format lmdb only.",
    )
    return parser


def _compress_safetensors_store(args, store_format: str) -> int:  # noqa: ANN001
    """Compress a ``.safetensors`` input into an LMDB or WebDataset store."""
    from ptwm.integrations import _compress_safetensors as _cs  # noqa: PLC0415

    input_path = Path(args.path)
    tensors, _total = _cs._read_safetensors(input_path)
    method_hint = _method_hint_from_name(getattr(args, "method", "MICROSCALE"))

    if store_format == "lmdb":
        from ptwm.stores import write_lmdb  # noqa: PLC0415

        out_path = (
            Path(args.out)
            if args.out
            else input_path.with_suffix(input_path.suffix + ".lmdb")
        )
        map_size = (
            _parse_size(args.lmdb_map_size)
            if getattr(args, "lmdb_map_size", None)
            else 8 * 1024**3
        )
        print(f"Compressing {input_path} → {out_path} (lmdb/{args.lmdb_layout})")  # noqa: T201
        write_lmdb(
            out_path,
            tensors,
            layout=args.lmdb_layout,
            map_size=map_size,
            method_hint=method_hint,
        )
    else:
        from ptwm.stores import write_webdataset  # noqa: PLC0415

        out_path = (
            Path(args.out)
            if args.out
            else input_path.with_suffix(input_path.suffix + ".wds")
        )
        max_shard = (
            _parse_size(args.store_max_shard_size)
            if getattr(args, "store_max_shard_size", None)
            else 2 * 1024**3
        )
        print(f"Compressing {input_path} → {out_path} (webdataset)")  # noqa: T201
        write_webdataset(
            out_path, tensors, max_shard_size=max_shard, method_hint=method_hint
        )
    return 0


def _method_hint_from_name(name: str) -> int:
    """Map a ``--method`` name to the wire ``method_hint`` integer."""
    return {
        "HUFFMAN": 1,
        "ZSTD": 2,
        "MICROSCALE": 3,
        "RANS": 4,
        "IDENTITY": 5,
        "AUTO": 3,
    }.get(str(name).upper(), 3)


def _compress_safetensors_integration(args, mode: str) -> int:  # noqa: ANN001
    """Compress a .safetensors input via the multi-tensor integration path.

    Used whenever --explore or --mode (a|b) is set. The legacy per-tensor
    path cannot explore; selecting any mode other than 'legacy' routes here.
    """
    from ptwm.classify import (  # noqa: PLC0415
        ClassifierChain,
        HeuristicClassifier,
    )
    from ptwm.integrations import (  # noqa: PLC0415
        compress_safetensors_file as _integrations_compress,
    )

    input_path = Path(args.path)
    explore_options = _explore_options_from_args(args)

    # Mode 'a' writes a directory; mode 'b' writes a single .safetensors-
    # shaped file. Output naming follows pre-existing usage.
    if mode == "a":
        out_path = (
            Path(args.out)
            if args.out
            else input_path.with_suffix(input_path.suffix + ".ptwm")
        )
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)
        is_existing_non_empty = any(out_dir.iterdir())
    else:
        out_path = (
            Path(args.out) if args.out else input_path.with_suffix(".ptwm.safetensors")
        )
        out_dir = out_path.parent / (out_path.stem + "__staging")
        out_dir.mkdir(parents=True, exist_ok=True)
        is_existing_non_empty = out_path.exists()

    if not args.force and is_existing_non_empty:
        user_input = (
            input(f"{out_path} already exists; overwrite (y/n)? ").strip().lower()
        )
        if user_input not in ("yes", "y"):
            return 0

    input_bytes = input_path.stat().st_size
    if explore_options is not None:
        print(  # noqa: T201
            f"Compressing {input_path} → {out_path} "
            f"(--explore budget {args.explore_budget_ms}ms per group)"
        )
    else:
        print(f"Compressing {input_path} → {out_path}")  # noqa: T201

    audit = _integrations_compress(
        input_path,
        out_dir,
        mode=mode,
        classifier=ClassifierChain([HeuristicClassifier()]),
        explore_options=explore_options,
        use_user_cache=not getattr(args, "no_cache", False),
    )

    output_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    if explore_options is not None:
        _print_explore_summary(audit, input_bytes, output_bytes)
    return 0


def _resolve_compress_mode(args) -> str | None:  # noqa: ANN001
    """Pick the effective compress mode for a .safetensors input.

    Returns 'a' / 'b' to route through the integration path, 'legacy' to use
    the per-tensor shell, or `None` for non-.safetensors inputs.
    """
    explicit = getattr(args, "ptwm_mode", None)
    if explicit is not None:
        return explicit
    if getattr(args, "explore", False):
        return "a"
    # No explicit mode and no --explore: keep legacy behaviour so existing
    # user workflows do not silently switch output shapes.
    return "legacy"


def handle_compress(args):
    check_and_install_ptwm()
    codec_menu = _parse_codec_menu(args.codec_menu)
    path = Path(args.path)
    if getattr(args, "explore", False):
        if not (path.is_file() and path.suffix == ".safetensors"):
            print(  # noqa: T201
                f"{RED}Error: --explore requires a .safetensors file input "
                f"(got: {args.path}).{RESET}",
                file=sys.stderr,
            )
            sys.exit(2)
        mode = _resolve_compress_mode(args)
        if mode == "legacy":
            print(  # noqa: T201
                f"{RED}Error: --explore is incompatible with --mode legacy. "
                f"Use --mode a (default) or --mode b.{RESET}",
                file=sys.stderr,
            )
            sys.exit(2)
        _compress_safetensors_integration(args, mode)
        return
    if path.is_file() and path.suffix == ".safetensors":
        store_format = getattr(args, "store_format", "ptwm")
        if store_format in {"lmdb", "webdataset"}:
            _compress_safetensors_store(args, store_format)
            return
        mode = _resolve_compress_mode(args)
        if mode in {"a", "b"}:
            _compress_safetensors_integration(args, mode)
            return
    if path.is_dir():
        suffix = ".safetensors"
        compress_path(
            suffix=suffix,
            dtype=args.dtype,
            streaming_chunk_size=args.streaming_chunk_size,
            path=args.path,
            delete=args.delete,
            recursive=args.recursive,
            force=args.force,
            max_processes=args.max_processes,
            hf_cache=args.hf_cache,
            model=args.model,
            branch=args.model_branch,
            method=args.method,
            verification=args.verification,
            test=args.test,
            is_streaming=args.is_streaming,
            threads=args.threads,
            file_compression=args.file_compression,
            codec_menu=codec_menu,
            codec=args.codec,
            device=args.device,
        )
    elif path.is_file():
        if args.delta:
            compress_file_delta(
                input_file=args.path,
                delta_file=args.delta,
                dtype=args.dtype,
                streaming_chunk_size=args.streaming_chunk_size,
                delete=args.delete,
                force=args.force,
                hf_cache=args.hf_cache,
                method=args.method,
                verification=args.verification,
                test=args.test,
                is_streaming=args.is_streaming,
                threads=args.threads,
                codec_menu=codec_menu,
                codec=args.codec,
                device=args.device,
            )
        elif path.suffix == ".safetensors" and not args.file_compression:
            compress_safetensors_file(
                filename=args.path,
                delete=args.delete,
                force=args.force,
                hf_cache=args.hf_cache,
                method=args.method,
                threads=args.threads,
                codec_menu=codec_menu,
                codec=args.codec,
                device=args.device,
            )
        else:
            compress_file(
                input_file=args.path,
                dtype=args.dtype,
                streaming_chunk_size=args.streaming_chunk_size,
                delete=args.delete,
                force=args.force,
                hf_cache=args.hf_cache,
                method=args.method,
                verification=args.verification,
                test=args.test,
                is_streaming=args.is_streaming,
                threads=args.threads,
                codec_menu=codec_menu,
                codec=args.codec,
                device=args.device,
            )
    elif args.hf_cache and args.model:
        compress_path(
            suffix=args.path,
            dtype=args.dtype,
            streaming_chunk_size=args.streaming_chunk_size,
            path=".",
            delete=args.delete,
            recursive=args.recursive,
            force=args.force,
            max_processes=args.max_processes,
            hf_cache=args.hf_cache,
            model=args.model,
            branch=args.model_branch,
            method=args.method,
            verification=args.verification,
            test=args.test,
            is_streaming=args.is_streaming,
            threads=args.threads,
            file_compression=args.file_compression,
            codec_menu=codec_menu,
            codec=args.codec,
            device=args.device,
        )
    else:
        print(  # noqa: T201
            f"{RED}Error: Path or suffix '{args.path}' not found.{RESET}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
