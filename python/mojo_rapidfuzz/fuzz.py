from __future__ import annotations

from typing import NamedTuple

from ._common import (
    cutoff_similarity,
    partial_ratio_raw,
    processed_pair,
    ratio_raw,
)


class ScoreAlignment(NamedTuple):
    score: float
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int


def _valid(first, second) -> bool:
    return first is not None and second is not None


def ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    return cutoff_similarity(ratio_raw(s1, s2), score_cutoff)


def partial_ratio_alignment(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return None
    s1, s2 = processed_pair(s1, s2, processor)
    score, (src_start, src_end, dest_start, dest_end) = partial_ratio_raw(s1, s2)
    if score_cutoff is not None and score < score_cutoff:
        return None
    return ScoreAlignment(score, src_start, src_end, dest_start, dest_end)


def partial_ratio(s1, s2, *, processor=None, score_cutoff=None):
    alignment = partial_ratio_alignment(
        s1, s2, processor=processor, score_cutoff=score_cutoff
    )
    return 0.0 if alignment is None else alignment.score


def _tokens(value):
    if not isinstance(value, (str, bytes)):
        raise TypeError("token scorers require strings or bytes")
    return value.split()


def _join(tokens, original):
    separator = b" " if isinstance(original, bytes) else " "
    return separator.join(tokens)


def token_sort_ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    score = ratio_raw(_join(sorted(_tokens(s1)), s1), _join(sorted(_tokens(s2)), s2))
    return cutoff_similarity(score, score_cutoff)


def _set_parts(s1, s2):
    left, right = set(_tokens(s1)), set(_tokens(s2))
    common = left & right
    return common, left - common, right - common


def token_set_ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    common, left, right = _set_parts(s1, s2)
    if not (common or left) or not (common or right):
        return 0.0
    if common and (not left or not right):
        return cutoff_similarity(100.0, score_cutoff)
    section = _join(sorted(common), s1)
    left_joined = _join(sorted(left), s1)
    right_joined = _join(sorted(right), s2)
    left_full = _join([part for part in (section, left_joined) if part], s1)
    right_full = _join([part for part in (section, right_joined) if part], s2)
    score = max(
        ratio_raw(left_full, right_full),
        ratio_raw(section, left_full),
        ratio_raw(section, right_full),
    )
    return cutoff_similarity(score, score_cutoff)


def token_ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    score = max(token_set_ratio(s1, s2), token_sort_ratio(s1, s2))
    return cutoff_similarity(score, score_cutoff)


def partial_token_sort_ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    score = partial_ratio(
        _join(sorted(_tokens(s1)), s1), _join(sorted(_tokens(s2)), s2)
    )
    return cutoff_similarity(score, score_cutoff)


def partial_token_set_ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    common, left, right = _set_parts(s1, s2)
    if not (common or left) or not (common or right):
        return 0.0
    if common:
        return cutoff_similarity(100.0, score_cutoff)
    score = partial_ratio(_join(sorted(left), s1), _join(sorted(right), s2))
    return cutoff_similarity(score, score_cutoff)


def partial_token_ratio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    score = max(
        partial_token_set_ratio(s1, s2), partial_token_sort_ratio(s1, s2)
    )
    return cutoff_similarity(score, score_cutoff)


def WRatio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    if not len(s1) or not len(s2):
        return 0.0
    base = ratio_raw(s1, s2)
    length_ratio = max(len(s1), len(s2)) / min(len(s1), len(s2))
    if length_ratio < 1.5:
        score = max(base, 0.95 * token_ratio(s1, s2))
    else:
        partial_scale = 0.9 if length_ratio <= 8 else 0.6
        score = max(
            base,
            partial_scale * partial_ratio(s1, s2),
            0.95 * partial_scale * partial_token_ratio(s1, s2),
        )
    return cutoff_similarity(score, score_cutoff)


def QRatio(s1, s2, *, processor=None, score_cutoff=None):
    if not _valid(s1, s2):
        return 0.0
    s1, s2 = processed_pair(s1, s2, processor)
    if not len(s1) or not len(s2):
        return 0.0
    return cutoff_similarity(ratio_raw(s1, s2), score_cutoff)


for _function in (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio,
    token_ratio,
    partial_token_sort_ratio,
    partial_token_set_ratio,
    partial_token_ratio,
    WRatio,
    QRatio,
):
    _function._score_kind = "similarity"
