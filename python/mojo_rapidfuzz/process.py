from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np

from . import fuzz
from ._common import ratio_many_raw, ratio_matrix_raw


def _entries(choices):
    return choices.items() if isinstance(choices, Mapping) else enumerate(choices)


def _ratio_results(query, choices, processor, score_cutoff):
    processed_query = processor(query) if processor is not None else query
    entries = [
        (key, choice)
        for key, choice in _entries(choices)
        if choice is not None
    ]
    processed = [
        processor(choice) if processor is not None else choice
        for _, choice in entries
    ]
    scores = ratio_many_raw(processed_query, processed)
    return [
        (choice, float(score), key)
        for (key, choice), score in zip(entries, scores)
        if score_cutoff is None or score >= score_cutoff
    ]


def _score(query, choice, scorer, scorer_kwargs, score_cutoff, score_hint):
    kwargs = dict(scorer_kwargs or {})
    if score_cutoff is not None:
        kwargs["score_cutoff"] = score_cutoff
    if score_hint is not None:
        try:
            parameters = inspect.signature(scorer).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        if any(
            parameter.name == "score_hint"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            kwargs["score_hint"] = score_hint
    return scorer(query, choice, **kwargs)


def extract_iter(
    query,
    choices,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    processed_query = processor(query) if processor is not None else query
    distance = getattr(scorer, "_score_kind", "similarity") == "distance"
    for key, choice in _entries(choices):
        if choice is None:
            continue
        candidate = processor(choice) if processor is not None else choice
        score = _score(
            processed_query,
            candidate,
            scorer,
            scorer_kwargs,
            score_cutoff,
            score_hint,
        )
        accepted = (
            score_cutoff is None
            or (score <= score_cutoff if distance else score >= score_cutoff)
        )
        if accepted:
            yield choice, score, key


def extractOne(
    query,
    choices,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    if scorer is fuzz.ratio and not scorer_kwargs:
        processed_query = processor(query) if processor is not None else query
        entries = [
            (key, choice)
            for key, choice in _entries(choices)
            if choice is not None
        ]
        if not entries:
            return None
        processed = [
            processor(choice) if processor is not None else choice
            for _, choice in entries
        ]
        scores = ratio_many_raw(processed_query, processed)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if score_cutoff is not None and best_score < score_cutoff:
            return None
        key, choice = entries[best_index]
        return choice, best_score, key
    distance = getattr(scorer, "_score_kind", "similarity") == "distance"
    best = None
    for result in extract_iter(
        query,
        choices,
        scorer=scorer,
        processor=processor,
        score_cutoff=score_cutoff,
        score_hint=score_hint,
        scorer_kwargs=scorer_kwargs,
    ):
        if best is None or (result[1] < best[1] if distance else result[1] > best[1]):
            best = result
            if (distance and result[1] == 0) or (
                not distance and result[1] == 100
            ):
                break
    return best


def extract(
    query,
    choices,
    *,
    scorer=fuzz.WRatio,
    processor=None,
    limit=5,
    score_cutoff=None,
    score_hint=None,
    scorer_kwargs=None,
):
    if scorer is fuzz.ratio and not scorer_kwargs:
        results = _ratio_results(query, choices, processor, score_cutoff)
    else:
        results = list(
            extract_iter(
                query,
                choices,
                scorer=scorer,
                processor=processor,
                score_cutoff=score_cutoff,
                score_hint=score_hint,
                scorer_kwargs=scorer_kwargs,
            )
        )
    reverse = getattr(scorer, "_score_kind", "similarity") != "distance"
    results.sort(key=lambda result: result[1], reverse=reverse)
    return results if limit is None else results[:limit]


def cdist(
    queries,
    choices,
    *,
    scorer=fuzz.ratio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    score_multiplier=1,
    dtype=None,
    workers=1,
    device="cpu",
    **kwargs,
):
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    queries = list(queries)
    choices = list(choices)
    processed_choices = [
        processor(choice) if processor is not None else choice for choice in choices
    ]
    if scorer is fuzz.ratio and not kwargs:
        processed_queries = [
            processor(query) if processor is not None else query for query in queries
        ]
        matrix = ratio_matrix_raw(
            processed_queries, processed_choices, device=device
        )
        if matrix is not None:
            if score_cutoff is not None:
                matrix[matrix < score_cutoff] = 0
            if dtype is None:
                dtype = np.float32
            return np.asarray(matrix * score_multiplier, dtype=dtype)
    rows = []
    for query in queries:
        query = processor(query) if processor is not None else query
        if scorer is fuzz.ratio and not kwargs:
            scores = ratio_many_raw(query, processed_choices)
            if score_cutoff is not None:
                scores[scores < score_cutoff] = 0
        else:
            scores = np.asarray(
                [
                    _score(
                        query,
                        choice,
                        scorer,
                        kwargs,
                        score_cutoff,
                        score_hint,
                    )
                    for choice in processed_choices
                ]
            )
        rows.append(scores * score_multiplier)
    if dtype is None:
        dtype = np.float32
    return np.asarray(rows, dtype=dtype).reshape(len(queries), len(choices))


def cpdist(
    queries,
    choices,
    *,
    scorer=fuzz.ratio,
    processor=None,
    score_cutoff=None,
    score_hint=None,
    score_multiplier=1,
    dtype=None,
    workers=1,
    **kwargs,
):
    queries = list(queries)
    choices = list(choices)
    if len(queries) != len(choices):
        raise ValueError("Length of queries and choices must be the same!")
    values = [
        _score(
            processor(query) if processor is not None else query,
            processor(choice) if processor is not None else choice,
            scorer,
            kwargs,
            score_cutoff,
            score_hint,
        )
        * score_multiplier
        for query, choice in zip(queries, choices)
    ]
    return np.asarray(values, dtype=np.float32 if dtype is None else dtype)
