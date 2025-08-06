"""PTWM ablation harness: resolved policy, bench-compress orchestration, attribution.

* _resolve.py:   active-trust-aware policy resolution
* _bench.py:     ptwm bench compress
* _compare.py:   ptwm bench compare
* _attribute.py: leave-one-out + Shapley
* _replay.py:    resolved-policy round-trip
"""

from ptwm.ablation._attribute import (
    AttributionResult,
    AttributionRow,
    attribute_leave_one_out,
    attribute_shapley,
)
from ptwm.ablation._bench import BenchResult, BenchRow, bench_compress
from ptwm.ablation._compare import CompareResult, bench_compare
from ptwm.ablation._replay import ReplayResult, bench_replay
from ptwm.ablation._resolve import resolve_active

__all__ = [
    "AttributionResult",
    "AttributionRow",
    "BenchResult",
    "BenchRow",
    "CompareResult",
    "ReplayResult",
    "attribute_leave_one_out",
    "attribute_shapley",
    "bench_compare",
    "bench_compress",
    "bench_replay",
    "resolve_active",
]
