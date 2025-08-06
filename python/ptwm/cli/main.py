import argparse
import sys

import ptwm
from ptwm.cli import bench, chains, compress, decompress, ext, policy, trust


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="ptwm", description="PTWM CLI")
    parser.add_argument(
        "--version",
        action="version",
        version=f"ptwm {ptwm.__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress.add_compress_parser(subparsers)
    decompress.add_decompress_parser(subparsers)
    chains.add_chains_parser(subparsers)
    trust.add_trust_parser(subparsers)
    policy.add_policy_parser(subparsers)
    ext.add_ext_parser(subparsers)
    bench.add_bench_parser(subparsers)

    parsed_args = parser.parse_args(args)
    if parsed_args.command == "compress":
        compress.handle_compress(parsed_args)
    elif parsed_args.command == "decompress":
        decompress.handle_decompress(parsed_args)
    elif parsed_args.command == "chains":
        parsed_args.func(parsed_args)
    elif parsed_args.command in {"trust", "policy", "ext", "bench"}:
        sys.exit(parsed_args.func(parsed_args))


if __name__ == "__main__":
    main()
