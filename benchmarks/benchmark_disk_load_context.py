# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Measure what `disk_load_context` saves on grouped reads from one safetensors file.

`safe_open` parses a header describing every tensor in the file, so the cost of
opening scales with how many tensors the file holds. Opening once per read makes
a group of N reads from one shard quadratic; holding the handle makes it linear.

Run:
    python benchmarks/benchmark_disk_load_context.py
"""

import gc
import os
import statistics
import tempfile
import time

import torch
from compressed_tensors.offload.cache.disk import _opened, disk_load_context
from safetensors.torch import save_file


REPS = 9


def _read_all(path: str, names: list[str]) -> None:
    for name in names:
        with _opened(path, "cpu") as file:
            file.get_tensor(name)


def _median(fn, *args) -> float:
    times = []
    for _ in range(REPS):
        gc.collect()
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run(num_tensors: int, numel: int = 8) -> None:
    names = [f"w{i}" for i in range(num_tensors)]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "shard.safetensors")
        save_file({n: torch.zeros(numel) for n in names}, path)

        _read_all(path, names)  # warm the page cache

        per_open = _median(_read_all, path, names)

        def grouped():
            with disk_load_context():
                _read_all(path, names)

        grouped_time = _median(grouped)

    per_read_us = (per_open - grouped_time) / num_tensors * 1e6
    print(
        f"  {num_tensors:5d}  {per_open * 1e3:10.2f}  {grouped_time * 1e3:10.2f}  "
        f"{per_open / grouped_time:7.1f}x  {per_read_us:9.1f}"
    )


def main() -> None:
    print("Grouped safetensors reads from a single shard, warm page cache")
    print(f"median of {REPS}, torch {torch.__version__}\n")
    print(
        f"  {'tensors':>5}  {'per-open ms':>10}  {'grouped ms':>10}  "
        f"{'speedup':>8}  {'us/open':>9}"
    )
    for num_tensors in (8, 32, 128, 384, 768):
        run(num_tensors)
    print(
        "\nPer-open cost grows with the tensor count in the file, so reading a "
        "group\nof N tensors one open at a time is quadratic. Holding the handle "
        "is linear."
    )


if __name__ == "__main__":
    main()
