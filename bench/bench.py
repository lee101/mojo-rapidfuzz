"""Benchmarks against upstream RapidFuzz on the same inputs."""

from __future__ import annotations

import os
import platform
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_rapidfuzz import fuzz, process  # noqa: E402
from mojo_rapidfuzz._common import _gpu_memory_available  # noqa: E402
from mojo_rapidfuzz.distance import Levenshtein  # noqa: E402
from rapidfuzz import fuzz as rf_fuzz  # noqa: E402
from rapidfuzz import process as rf_process  # noqa: E402
from rapidfuzz.distance import Levenshtein as RFLevenshtein  # noqa: E402


def timed(function, number, repeat=7):
    samples = []
    function()
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            function()
        samples.append((time.perf_counter() - start) / number)
    return statistics.median(samples)


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = random.Random(19)
    a = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=256))
    b = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=256))
    short = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz ", k=40))
    long = "prefix " + short[:28] + "".join(rng.choices("abcdef ", k=52))
    choices = [
        "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=24))
        for _ in range(10_000)
    ]
    queries = choices[:100]
    matrix_choices = choices[100:300]

    cases = [
        (
            "Levenshtein.distance (256 chars)",
            lambda: Levenshtein.distance(a, b),
            lambda: RFLevenshtein.distance(a, b),
            500,
        ),
        (
            "fuzz.ratio (256 chars)",
            lambda: fuzz.ratio(a, b),
            lambda: rf_fuzz.ratio(a, b),
            500,
        ),
        (
            "fuzz.partial_ratio (40 vs 87)",
            lambda: fuzz.partial_ratio(short, long),
            lambda: rf_fuzz.partial_ratio(short, long),
            100,
        ),
        (
            "process.extractOne ratio (10k)",
            lambda: process.extractOne(a[:24], choices, scorer=fuzz.ratio),
            lambda: rf_process.extractOne(a[:24], choices, scorer=rf_fuzz.ratio),
            5,
        ),
        (
            "process.cdist ratio (100x200)",
            lambda: process.cdist(queries, matrix_choices, scorer=fuzz.ratio),
            lambda: rf_process.cdist(
                queries, matrix_choices, scorer=rf_fuzz.ratio
            ),
            1,
        ),
    ]

    print(f"Machine: {machine()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-rapidfuzz | rapidfuzz | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs, number in cases:
        mojo_time = timed(ours, number)
        reference_time = timed(theirs, number)
        ratio = reference_time / mojo_time
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{reference_time * 1e3:.3f} ms | {ratio:.2f}x {label} |"
        )

    if os.environ.get("MRF_BENCH_GPU") == "1":
        if not _gpu_memory_available():
            print()
            print("GPU benchmark skipped: less than 4000 MiB is free.")
            return
        gpu_queries = [
            "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=32))
            for _ in range(1024)
        ]
        gpu_choices = [
            "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=32))
            for _ in range(1024)
        ]
        cpu_time = timed(
            lambda: process.cdist(
                gpu_queries, gpu_choices, scorer=fuzz.ratio, device="cpu"
            ),
            1,
            repeat=3,
        )
        gpu_time = timed(
            lambda: process.cdist(
                gpu_queries, gpu_choices, scorer=fuzz.ratio, device="gpu"
            ),
            1,
            repeat=3,
        )
        print()
        print("| optional device case | CPU | GPU | result |")
        print("| --- | ---: | ---: | ---: |")
        print(
            "| process.cdist ratio (1024x1024, 32 chars) | "
            f"{cpu_time * 1e3:.3f} ms | {gpu_time * 1e3:.3f} ms | "
            f"{cpu_time / gpu_time:.2f}x GPU speedup |"
        )


if __name__ == "__main__":
    main()
