import collections
import contextlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ptwm.cli.utils import (
    RED,
    RESET,
    check_and_install_ptwm,
    replace_in_file,
)


def decompress_file(
    input_file: str,
    delete: bool = False,
    force: bool = False,
    hf_cache: bool = False,
    threads: int | None = None,
    quiet: bool = False,
) -> None:
    """Decompress a single file."""
    from ptwm import DecompressionConfig, Decompressor  # noqa: PLC0415

    full_path = Path(input_file)
    if not str(input_file).endswith(".ptwm"):
        if not quiet:
            print(  # noqa: T201
                f"{RED}Input file does not have a '.ptwm' suffix{RESET}",
                file=sys.stderr,
            )
        return

    if full_path.exists():
        decompressed_path = full_path.with_suffix("")
        if not force and decompressed_path.exists():
            user_input = (
                input(f"{decompressed_path} already exists; overwrite (y/n)? ")
                .strip()
                .lower()
            )
            if user_input not in ("yes", "y"):
                return

        output_file = decompressed_path
        decompressor = Decompressor(DecompressionConfig(threads=threads))

        with full_path.open("rb") as infile, output_file.open("wb") as outfile:
            chunk = infile.read()
            d_data = decompressor.decompress(chunk)
            outfile.write(d_data)

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
            except Exception as e:
                msg = f"Error reorganizing Hugging Face cache: {e}"
                raise RuntimeError(msg) from e


def decompress_file_delta(
    input_file: str,
    delta_file: str,
    delete: bool = False,
    force: bool = False,
    hf_cache: bool = False,
    threads: int | None = None,
) -> None:
    """Decompress a file using delta compression."""
    from ptwm import DecompressionConfig, Decompressor  # noqa: PLC0415

    full_path = Path(input_file)
    delta_path = Path(delta_file)

    if not str(input_file).endswith(".ptwm"):
        print(  # noqa: T201
            f"{RED}Input file does not have the '.ptwm' suffix{RESET}", file=sys.stderr
        )
        return

    if full_path.exists() and delta_path.exists():
        if delete and not hf_cache:
            msg = f"{RED}Delete not supported yet for delta decompression.{RESET}"
            raise ValueError(msg)

        decompressed_path = full_path.with_suffix("")
        if not force and decompressed_path.exists():
            user_input = (
                input(f"{decompressed_path} already exists; overwrite (y/n)? ")
                .strip()
                .lower()
            )
            if user_input not in ("yes", "y"):
                return

        if "_delta_" in full_path.name:
            output_file = full_path.parent / (
                full_path.name.split("_delta_")[0] + ".bin"
            )
        else:
            output_file = decompressed_path

        decompressor = Decompressor(
            DecompressionConfig(
                delta_second_data=delta_path.read_bytes(),
                threads=threads,
            )
        )

        with full_path.open("rb") as f:
            file_data = f.read()
        decompressed_data = decompressor.decompress(file_data)
        with output_file.open("wb") as f_out:
            f_out.write(decompressed_data)

        if hf_cache:
            try:
                snapshot_path = full_path.parent
                blob_name = snapshot_path / full_path.readlink()
                output_file.rename(blob_name)
                output_file.symlink_to(blob_name)
                if full_path.exists():
                    full_path.unlink()
            except Exception as e:
                msg = f"Error reorganizing Hugging Face cache: {e}"
                raise RuntimeError(msg) from e


def decompress_safetensors_file(
    filename: str,
    delete: bool = False,
    force: bool = False,
    hf_cache: bool = False,
    threads: int | None = None,
    quiet: bool = False,
) -> None:
    """Decompress a safetensors file."""
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    from ptwm import DecompressionConfig, Decompressor  # noqa: PLC0415
    from ptwm.utils._safetensors import (  # noqa: PLC0415
        get_compressed_tensors_metadata,
    )

    full_path = Path(filename)
    if not str(filename).endswith("ptwm.safetensors"):
        if not quiet:
            print(  # noqa: T201
                f"{RED}File {filename} is not a .ptwm.safetensors file.{RESET}",
                file=sys.stderr,
            )
        return

    decompressed_path = full_path.parent / (
        full_path.name[: -(len(".ptwm.safetensors"))] + ".safetensors"
    )
    if not force and decompressed_path.exists():
        user_input = (
            input(f"{decompressed_path} already exists; overwrite (y/n)? ")
            .strip()
            .lower()
        )
        if user_input not in ("yes", "y"):
            return

    tensors = {}
    decompressor = Decompressor(DecompressionConfig(threads=threads))
    with safe_open(filename, "pt", "cpu") as f:
        metadata_raw = f.metadata()
        compressed_metadata = get_compressed_tensors_metadata(metadata_raw)
        for name in f.keys():  # noqa: SIM118
            tensor = f.get_tensor(name)
            if name not in compressed_metadata:
                tensors[name] = tensor
                continue

            decompressed_buf = decompressor.decompress(tensor.contiguous().numpy())
            tensors[name] = decompressed_buf

        metadata = f.metadata()
        if metadata:
            metadata.pop("ptwm_compressed_vectors", None)

    save_file(tensors, str(decompressed_path), metadata)

    if delete and not hf_cache:
        full_path.unlink()

    if hf_cache:
        try:
            snapshot_path = full_path.parent
            blob_name = snapshot_path / full_path.readlink()
            decompressed_path.rename(blob_name)
            decompressed_path.symlink_to(blob_name)
            if full_path.exists():
                full_path.unlink()
        except Exception as e:
            msg = f"Error reorganizing Hugging Face cache: {e}"
            raise RuntimeError(msg) from e


def decompress_path(
    path: str = ".",
    delete: bool = False,
    force: bool = False,
    max_processes: int = 1,
    hf_cache: bool = False,
    model: str = "",
    branch: str = "main",
    threads: int | None = None,
) -> None:
    """Decompress every .ptwm file under ``path``."""
    overwrite_first = True
    file_list = []
    is_file_safetensors_compression = {}
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

    for full_path in search_path.glob("**/*"):
        if full_path.name.endswith((".ptwm", ".ptwm.safetensors")):
            if full_path.name.endswith(".ptwm.safetensors"):
                decompressed_name = (
                    full_path.name[: -(len(".ptwm.safetensors"))] + ".safetensors"
                )
                is_file_safetensors_compression[str(full_path)] = 1
            else:
                decompressed_name = full_path.name[: -len(".ptwm")]
                is_file_safetensors_compression[str(full_path)] = 0

            decompressed_path = full_path.parent / decompressed_name
            if not force and decompressed_path.exists():
                if overwrite_first:
                    overwrite_first = False
                    user_input = (
                        input(
                            "Decompressed files already exists; "
                            "Would you like to overwrite them all (y/n)? "
                        )
                        .strip()
                        .lower()
                    )
                    if user_input in ("y", "yes"):
                        force = True

                if not force:
                    user_input = (
                        input(f"{decompressed_path} already exists; overwrite (y/n)? ")
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

        for file_name in file_list:
            old = (
                "ptwm.safetensors"
                if is_file_safetensors_compression[file_name] == 1
                else "safetensors.ptwm"
            )
            new = "safetensors"

            index_path_safe = search_path / SAFE_WEIGHTS_INDEX_NAME
            index_path_weights = search_path / WEIGHTS_INDEX_NAME

            if index_path_safe.exists():
                blob_path = search_path / index_path_safe.readlink()
                replace_in_file(file_path=blob_path, old=old, new=new)
            elif index_path_weights.exists():
                blob_path = search_path / index_path_weights.readlink()
                replace_in_file(file_path=blob_path, old=old, new=new)

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        future_to_file = {}
        for file in file_list[:max_processes]:
            func = (
                decompress_safetensors_file
                if is_file_safetensors_compression[file] == 1
                else decompress_file
            )
            future_to_file[
                executor.submit(func, file, delete, True, hf_cache, threads, True)
            ] = file

        remaining_files = collections.deque(file_list[max_processes:])
        while future_to_file:
            for future in as_completed(future_to_file):
                future_to_file.pop(future)
                with contextlib.suppress(Exception):
                    future.result()
                if remaining_files:
                    next_file = remaining_files.popleft()
                    func = (
                        decompress_safetensors_file
                        if is_file_safetensors_compression[next_file] == 1
                        else decompress_file
                    )
                    future_to_file[
                        executor.submit(
                            func, next_file, delete, True, hf_cache, threads, True
                        )
                    ] = next_file


def add_decompress_parser(subparsers):
    parser = subparsers.add_parser("decompress", help="Decompress a file or directory")
    parser.add_argument(
        "path", type=str, nargs="?", default="", help="Path to file or directory"
    )
    parser.add_argument("--delta", type=str, help="Path to delta file")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete compressed file after decompression",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting when decompressing.",
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="Indicate if the file is in the Hugging Face cache.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="The amount of threads to be used.",
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
    return parser


def _decompress_store(path: Path, store_format: str, *, force: bool) -> None:
    """Reconstruct a ``.safetensors`` file from an LMDB / WebDataset store."""
    import safetensors.torch as st  # noqa: PLC0415

    from ptwm.stores import open_store  # noqa: PLC0415

    store = open_store(path)
    state = {name: store.get_tensor(name) for name in store.names()}
    suffix = ".lmdb" if store_format == "lmdb" else ".wds"
    base = path.name.removesuffix(suffix)
    out_path = path.parent / f"{base}.safetensors"
    if out_path.exists() and not force:
        user_input = (
            input(f"{out_path} already exists; overwrite (y/n)? ").strip().lower()
        )
        if user_input not in ("yes", "y"):
            return
    st.save_file(state, str(out_path))
    print(f"Decompressed {path} → {out_path} ({store_format})")  # noqa: T201


def handle_decompress(args):
    check_and_install_ptwm()
    path_str = args.path or "."
    path = Path(path_str)

    if path.exists():
        from ptwm.stores import detect_format  # noqa: PLC0415

        store_format = detect_format(path)
        if store_format in {"lmdb", "webdataset"}:
            _decompress_store(path, store_format, force=args.force)
            return

    if path.is_dir() or not args.path:
        decompress_path(
            path=path_str,
            delete=args.delete,
            force=args.force,
            max_processes=args.max_processes,
            hf_cache=args.hf_cache,
            model=args.model,
            branch=args.model_branch,
            threads=args.threads,
        )
    elif path.is_file():
        if args.delta:
            decompress_file_delta(
                input_file=path_str,
                delta_file=args.delta,
                delete=args.delete,
                force=args.force,
                hf_cache=args.hf_cache,
                threads=args.threads,
            )
        elif path.name.endswith(".ptwm.safetensors"):
            decompress_safetensors_file(
                filename=path_str,
                delete=args.delete,
                force=args.force,
                hf_cache=args.hf_cache,
                threads=args.threads,
            )
        else:
            decompress_file(
                input_file=path_str,
                delete=args.delete,
                force=args.force,
                hf_cache=args.hf_cache,
                threads=args.threads,
            )
    else:
        print(f"{RED}Error: Path '{path_str}' not found.{RESET}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
