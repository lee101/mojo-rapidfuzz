from __future__ import annotations

import subprocess
from collections.abc import Sequence
from functools import lru_cache
from threading import local
from typing import Any

import numpy as np

from ._lib import addr, lib

_scratch_buffers = local()


def _storage(values: list[int] | np.ndarray) -> tuple[np.ndarray, int]:
    size = len(values)
    array = np.empty(max(1, size), dtype=np.uint32)
    if size:
        array[:size] = values
    return array, size


def _string_codes(value: str | bytes | bytearray) -> tuple[np.ndarray, int]:
    if isinstance(value, str):
        if not value:
            return np.zeros(1, dtype=np.uint32), 0
        raw = value.encode("utf-32-le", "surrogatepass")
        return np.frombuffer(raw, dtype=np.uint32), len(value)
    return _storage(np.frombuffer(bytes(value), dtype=np.uint8).astype(np.uint32))


@lru_cache(maxsize=1024)
def _cached_ascii_codes(value: str) -> np.ndarray:
    if not value:
        return np.zeros(1, dtype=np.uint8)
    raw = value.encode("ascii")
    return np.frombuffer(raw, dtype=np.uint8)


def _ascii_codes(value: str) -> tuple[np.ndarray, int]:
    if len(value) <= 4096:
        return _cached_ascii_codes(value), len(value)
    raw = value.encode("ascii")
    return np.frombuffer(raw, dtype=np.uint8), len(value)


@lru_cache(maxsize=256)
def _cached_ascii_pattern(value: str) -> tuple[np.ndarray, np.ndarray]:
    codes = _cached_ascii_codes(value)
    words = max(1, (len(value) + 63) // 64)
    masks = np.zeros(256 * words, dtype=np.uint64)
    for position, code in enumerate(codes[: len(value)]):
        index = int(code) * words + position // 64
        masks[index] |= np.uint64(1) << np.uint64(position % 64)
    return codes, masks


def _ascii_pattern(value: str) -> tuple[np.ndarray, np.ndarray]:
    if len(value) <= 512:
        return _cached_ascii_pattern(value)
    codes, _ = _ascii_codes(value)
    words = max(1, (len(value) + 63) // 64)
    masks = np.zeros(256 * words, dtype=np.uint64)
    for position, code in enumerate(codes):
        index = int(code) * words + position // 64
        masks[index] |= np.uint64(1) << np.uint64(position % 64)
    return codes, masks


def _u64_scratch(size: int) -> np.ndarray:
    scratch = getattr(_scratch_buffers, "u64", None)
    if scratch is None or len(scratch) < size:
        scratch = np.empty(size, dtype=np.uint64)
        _scratch_buffers.u64 = scratch
    return scratch


def _ascii_collection(values: Sequence[str]):
    try:
        joined = "".join(values)
    except TypeError:
        return None
    if not joined.isascii():
        return None
    lengths = np.fromiter(map(len, values), dtype=np.int64, count=len(values))
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    raw = joined.encode("ascii")
    flat = (
        np.frombuffer(raw, dtype=np.uint8)
        if raw
        else np.zeros(1, dtype=np.uint8)
    )
    return flat, offsets


def encode_pair(first: Any, second: Any) -> tuple[np.ndarray, int, np.ndarray, int]:
    if isinstance(first, (str, bytes, bytearray)) and isinstance(
        second, (str, bytes, bytearray)
    ):
        return (*_string_codes(first), *_string_codes(second))
    try:
        left = list(first)
        right = list(second)
    except TypeError:
        len(first)
        raise
    symbols: dict[Any, int] = {}

    def encode(values: list[Any]) -> tuple[np.ndarray, int]:
        encoded: list[int] = []
        for value in values:
            try:
                code = symbols.get(value)
            except TypeError as exc:
                raise TypeError("sequence elements must be hashable") from exc
            if code is None:
                code = len(symbols) + 1
                symbols[value] = code
            encoded.append(code)
        return _storage(encoded)

    return (*encode(left), *encode(right))


def processed_pair(first: Any, second: Any, processor):
    if processor is not None:
        first = processor(first)
        second = processor(second)
    return first, second


def distance_raw(
    first: Any, second: Any, weights: tuple[int, int, int]
) -> tuple[int, int, int]:
    if (
        weights == (1, 1, 1)
        and isinstance(first, str)
        and isinstance(second, str)
        and first.isascii()
        and second.isascii()
    ):
        if len(first) <= len(second):
            pattern, text = first, second
        else:
            pattern, text = second, first
        _, masks = _ascii_pattern(pattern)
        text_codes, text_len = _ascii_codes(text)
        n, m = len(first), len(second)
        words = max(1, (min(n, m) + 63) // 64)
        state = _u64_scratch(2 * words)
        distance = lib().mrf_levenshtein_distance_ascii_masked(
            addr(text_codes),
            len(pattern),
            text_len,
            addr(masks),
            addr(state),
        )
        insert_cost, delete_cost, replace_cost = weights
        maximum = max(n, m)
        return int(distance), maximum, maximum
    a, n, b, m = encode_pair(first, second)
    work = np.empty(max(1, m + 1), dtype=np.int64)
    insert_cost, delete_cost, replace_cost = weights
    distance = lib().mrf_levenshtein_distance(
        addr(a),
        addr(b),
        n,
        m,
        insert_cost,
        delete_cost,
        replace_cost,
        addr(work),
    )
    maximum = min(
        n * delete_cost + m * insert_cost,
        min(n, m) * replace_cost
        + max(0, n - m) * delete_cost
        + max(0, m - n) * insert_cost,
    )
    return int(distance), int(maximum), max(n, m)


def ratio_raw(first: Any, second: Any) -> float:
    if (
        isinstance(first, str)
        and isinstance(second, str)
        and first.isascii()
        and second.isascii()
    ):
        if len(first) <= len(second):
            pattern, text = first, second
        else:
            pattern, text = second, first
        _, masks = _ascii_pattern(pattern)
        text_codes, text_len = _ascii_codes(text)
        n, m = len(first), len(second)
        words = max(1, (min(n, m) + 63) // 64)
        state = _u64_scratch(words)
        return float(
            lib().mrf_ratio_ascii_masked(
                addr(text_codes),
                len(pattern),
                text_len,
                addr(masks),
                addr(state),
            )
        )
    a, n, b, m = encode_pair(first, second)
    work = np.empty(max(1, m + 1), dtype=np.int64)
    return float(lib().mrf_ratio(addr(a), addr(b), n, m, addr(work)))


def partial_ratio_raw(
    first: Any, second: Any
) -> tuple[float, tuple[int, int, int, int]]:
    if (
        isinstance(first, str)
        and isinstance(second, str)
        and first.isascii()
        and second.isascii()
        and min(len(first), len(second)) <= 63
    ):
        a, n = _ascii_codes(first)
        b, m = _ascii_codes(second)
        scratch = _u64_scratch(257)
        alignment = np.zeros(4, dtype=np.int64)
        score = lib().mrf_partial_ratio_ascii(
            addr(a), addr(b), n, m, addr(scratch), addr(alignment)
        )
        return float(score), tuple(int(value) for value in alignment)
    a, n, b, m = encode_pair(first, second)
    work = np.empty(max(1, max(n, m) + 1), dtype=np.int64)
    alignment = np.zeros(4, dtype=np.int64)
    score = lib().mrf_partial_ratio(
        addr(a), addr(b), n, m, addr(work), addr(alignment)
    )
    return float(score), tuple(int(value) for value in alignment)


def ratio_many_raw(query: Any, choices: Sequence[Any]) -> np.ndarray:
    choices = list(choices)
    if (
        isinstance(query, str)
        and query.isascii()
        and len(query) <= 63
        and (packed := _ascii_collection(choices)) is not None
    ):
        query_array, query_len = _ascii_codes(query)
        flat, offsets = packed
        scores = np.empty(max(1, len(choices)), dtype=np.float64)
        masks = np.empty(256, dtype=np.uint64)
        if choices:
            lib().mrf_ratio_many_ascii(
                addr(query_array),
                query_len,
                addr(flat),
                addr(offsets),
                len(choices),
                addr(scores),
                addr(masks),
            )
        return scores[: len(choices)]

    query_values = list(query)
    symbols: dict[Any, int] = {}
    query_codes = []
    for item in query_values:
        try:
            code = symbols.get(item)
        except TypeError as exc:
            raise TypeError("sequence elements must be hashable") from exc
        if code is None:
            code = len(symbols) + 1
            symbols[item] = code
        query_codes.append(code)
    encoded = [query_codes]
    for value in choices:
        row = []
        for item in value:
            try:
                row.append(symbols.get(item, 0))
            except TypeError as exc:
                raise TypeError("sequence elements must be hashable") from exc
        encoded.append(row)

    query_array, query_len = _storage(encoded[0])
    masks = np.zeros(max(1, len(symbols) + 1), dtype=np.uint64)
    if query_len <= 63:
        for position, code in enumerate(query_codes):
            masks[code] |= np.uint64(1) << np.uint64(position)
    offsets = np.zeros(len(choices) + 1, dtype=np.int64)
    for index, row in enumerate(encoded[1:], 1):
        offsets[index] = offsets[index - 1] + len(row)
    flat, _ = _storage([item for row in encoded[1:] for item in row])
    scores = np.empty(max(1, len(choices)), dtype=np.float64)
    work = np.empty(
        max(1, max((len(row) for row in encoded[1:]), default=0) + 1),
        dtype=np.int64,
    )
    if choices:
        lib().mrf_ratio_many(
            addr(query_array),
            query_len,
            addr(flat),
            addr(offsets),
            len(choices),
            addr(work),
            addr(scores),
            addr(masks),
        )
    return scores[: len(choices)]


def ratio_matrix_raw(
    queries: Sequence[Any], choices: Sequence[Any], device: str = "cpu"
) -> np.ndarray | None:
    queries = list(queries)
    choices = list(choices)
    if any(
        not isinstance(query, str) or not query.isascii() or len(query) > 63
        for query in queries
    ):
        return None
    packed_queries = _ascii_collection(queries)
    packed_choices = _ascii_collection(choices)
    if packed_queries is None or packed_choices is None:
        return None
    query_flat, query_offsets = packed_queries
    choice_flat, choice_offsets = packed_choices
    scores = np.empty(max(1, len(queries) * len(choices)), dtype=np.float64)
    masks = np.empty(max(1, len(queries) * 256), dtype=np.uint64)
    if queries and choices:
        used_gpu = False
        if device == "gpu" and _gpu_memory_available():
            allocation_bytes = (
                query_flat.nbytes
                + query_offsets.nbytes
                + choice_flat.nbytes
                + choice_offsets.nbytes
                + scores.nbytes
                + masks.nbytes
            )
            if allocation_bytes < 2_000_000_000:
                used_gpu = bool(
                    lib().mrf_ratio_matrix_ascii_gpu(
                        addr(query_flat),
                        len(query_flat),
                        addr(query_offsets),
                        len(queries),
                        addr(choice_flat),
                        len(choice_flat),
                        addr(choice_offsets),
                        len(choices),
                        addr(scores),
                    )
                )
        if not used_gpu:
            lib().mrf_ratio_matrix_ascii(
                addr(query_flat),
                addr(query_offsets),
                len(queries),
                addr(choice_flat),
                addr(choice_offsets),
                len(choices),
                addr(scores),
                addr(masks),
            )
    return scores[: len(queries) * len(choices)].reshape(
        len(queries), len(choices)
    )


def _gpu_memory_available() -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        first_gpu = int(result.stdout.splitlines()[0].strip())
    except (FileNotFoundError, IndexError, OSError, subprocess.SubprocessError, ValueError):
        return False
    return first_gpu >= 4000


def cutoff_similarity(score: float, score_cutoff: float | None) -> float:
    return score if score_cutoff is None or score >= score_cutoff else 0.0
