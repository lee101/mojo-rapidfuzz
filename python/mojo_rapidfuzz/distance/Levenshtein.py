from __future__ import annotations

import operator

import numpy as np

from .._common import distance_raw, encode_pair, processed_pair
from .._lib import addr, lib
from ._initialize import Editop, Editops


def _weights(weights) -> tuple[int, int, int]:
    if weights is None:
        return (1, 1, 1)
    try:
        values = tuple(weights)
        result = tuple(
            int(value) if isinstance(value, float) else operator.index(value)
            for value in values
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("weights must be a tuple of three integers") from exc
    if len(result) != 3:
        raise ValueError("weights must have three entries")
    if any(value < 0 for value in result):
        raise ValueError("weights must be non-negative")
    if any(value > np.iinfo(np.int64).max for value in result):
        raise OverflowError("weights exceed the signed 64-bit Mojo ABI")
    return result


def distance(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    s1, s2 = processed_pair(s1, s2, processor)
    score, _, _ = distance_raw(s1, s2, _weights(weights))
    if score_cutoff is not None and score > score_cutoff:
        return int(score_cutoff) + 1
    return score


def similarity(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    s1, s2 = processed_pair(s1, s2, processor)
    score, maximum, _ = distance_raw(s1, s2, _weights(weights))
    result = maximum - score
    return result if score_cutoff is None or result >= score_cutoff else 0


def normalized_distance(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    s1, s2 = processed_pair(s1, s2, processor)
    score, maximum, _ = distance_raw(s1, s2, _weights(weights))
    result = score / maximum if maximum else 0.0
    return result if score_cutoff is None or result <= score_cutoff else 1.0


def normalized_similarity(
    s1,
    s2,
    *,
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
):
    s1, s2 = processed_pair(s1, s2, processor)
    score, maximum, _ = distance_raw(s1, s2, _weights(weights))
    result = 1.0 - score / maximum if maximum else 1.0
    return result if score_cutoff is None or result >= score_cutoff else 0.0


def editops(s1, s2, *, processor=None, score_hint=None):
    s1, s2 = processed_pair(s1, s2, processor)
    a, n, b, m = encode_pair(s1, s2)
    prefix = 0
    while prefix < n and prefix < m and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < n - prefix
        and suffix < m - prefix
        and a[n - suffix - 1] == b[m - suffix - 1]
    ):
        suffix += 1
    a_mid = np.ascontiguousarray(a[prefix : n - suffix], dtype=np.uint32)
    b_mid = np.ascontiguousarray(b[prefix : m - suffix], dtype=np.uint32)
    mid_n, mid_m = len(a_mid), len(b_mid)
    if not mid_n:
        a_mid = np.zeros(1, dtype=np.uint32)
    if not mid_m:
        b_mid = np.zeros(1, dtype=np.uint32)
    matrix = np.empty((mid_n + 1, mid_m + 1), dtype=np.int64)
    lib().mrf_levenshtein_matrix(
        addr(a_mid), addr(b_mid), mid_n, mid_m, addr(matrix)
    )
    operations = []
    i, j = mid_n, mid_m
    while i or j:
        if i and j and a_mid[i - 1] == b_mid[j - 1]:
            i -= 1
            j -= 1
        elif i and matrix[i, j] == matrix[i - 1, j] + 1:
            i -= 1
            operations.append(Editop("delete", prefix + i, prefix + j))
        elif i and j and matrix[i, j] == matrix[i - 1, j - 1] + 1:
            i -= 1
            j -= 1
            operations.append(Editop("replace", prefix + i, prefix + j))
        else:
            j -= 1
            operations.append(Editop("insert", prefix + i, prefix + j))
    operations.reverse()
    return Editops(operations, n, m)


def opcodes(s1, s2, *, processor=None, score_hint=None):
    return editops(s1, s2, processor=processor, score_hint=score_hint).as_opcodes()


distance._score_kind = "distance"
similarity._score_kind = "similarity"
normalized_distance._score_kind = "distance"
normalized_similarity._score_kind = "similarity"
