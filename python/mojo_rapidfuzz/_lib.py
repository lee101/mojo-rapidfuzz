"""ctypes bridge to the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "rapidfuzz.mojo")
LIB = os.environ.get("MOJO_RAPIDFUZZ_LIB") or os.path.join(
    ROOT, "dist", "libmojo-rapidfuzz.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mrf_levenshtein_distance": ([I] * 8, I),
    "mrf_levenshtein_distance_ascii": ([I] * 5, I),
    "mrf_levenshtein_distance_ascii_masked": ([I] * 5, I),
    "mrf_ratio": ([I] * 5, F),
    "mrf_ratio_ascii": ([I] * 5, F),
    "mrf_ratio_ascii_masked": ([I] * 5, F),
    "mrf_partial_ratio": ([I] * 6, F),
    "mrf_partial_ratio_ascii": ([I] * 6, F),
    "mrf_ratio_many": ([I] * 8, None),
    "mrf_ratio_many_ascii": ([I] * 7, None),
    "mrf_ratio_matrix_ascii": ([I] * 8, None),
    "mrf_ratio_matrix_ascii_gpu": ([I] * 9, I),
    "mrf_levenshtein_matrix": ([I] * 5, None),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and (not os.path.exists(SRC) or os.path.getmtime(LIB) >= os.path.getmtime(SRC))
    ):
        return LIB
    if os.environ.get("MOJO_RAPIDFUZZ_LIB"):
        raise RuntimeError(f"MOJO_RAPIDFUZZ_LIB does not exist: {LIB}")
    script = os.path.join(ROOT, "build", "build.sh")
    subprocess.run(["bash", script], cwd=ROOT, check=True, timeout=1800)
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    """Return an address only for arrays that satisfy the Mojo ABI contract."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in {
        np.dtype(np.uint8),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.int64),
        np.dtype(np.float64),
    }:
        raise TypeError(f"unsupported FFI buffer dtype: {array.dtype}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    address = int(array.ctypes.data)
    if not address:
        raise ValueError("FFI buffers must have a non-null address")
    return address


if __name__ == "__main__":
    print(build(force=True))
