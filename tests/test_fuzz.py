import random

import pytest
from rapidfuzz import fuzz as reference

from mojo_rapidfuzz import fuzz

SCORERS = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "token_ratio",
    "partial_token_sort_ratio",
    "partial_token_set_ratio",
    "partial_token_ratio",
    "WRatio",
    "QRatio",
]

PAIRS = [
    ("", ""),
    ("a", "ab"),
    ("ab", "acb"),
    ("abcd", "XabcY"),
    ("new york mets", "new york jets"),
    ("fuzzy was a bear", "fuzzy fuzzy was a bear"),
    ("this is a test", "this is a test!"),
    ("你好 世界", "世界 你好"),
]


@pytest.mark.parametrize("name", SCORERS)
def test_scorer_parity(name):
    ours = getattr(fuzz, name)
    theirs = getattr(reference, name)
    for left, right in PAIRS:
        assert ours(left, right) == pytest.approx(theirs(left, right))


def test_randomized_scorer_parity():
    rng = random.Random(11)
    for _ in range(100):
        left = "".join(rng.choices("abcd ", k=rng.randrange(18)))
        right = "".join(rng.choices("abcd ", k=rng.randrange(18)))
        for name in SCORERS:
            assert getattr(fuzz, name)(left, right) == pytest.approx(
                getattr(reference, name)(left, right)
            )


def test_cutoff_and_processor_parity():
    for name in SCORERS:
        ours = getattr(fuzz, name)
        theirs = getattr(reference, name)
        assert ours(
            "NEW YORK", "new york city", processor=str.lower, score_cutoff=80
        ) == pytest.approx(
            theirs(
                "NEW YORK", "new york city", processor=str.lower, score_cutoff=80
            )
        )


def test_none_matches_rapidfuzz_behavior():
    for name in SCORERS:
        assert getattr(fuzz, name)(None, "value") == 0


def test_partial_ratio_alignment_parity():
    for left, right in [("abcd", "XabcY"), ("ab", "ba"), ("spam", "park")]:
        assert fuzz.partial_ratio_alignment(
            left, right
        ) == reference.partial_ratio_alignment(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"\x00\xffabc", b"\x00\xfeabc"),
        ([1, 2, 3], [1, 4, 3, 5]),
        (("alpha", None), ("alpha", "beta")),
    ],
)
def test_base_ratio_sequence_parity(left, right):
    assert fuzz.ratio(left, right) == pytest.approx(reference.ratio(left, right))
    assert fuzz.partial_ratio(left, right) == pytest.approx(
        reference.partial_ratio(left, right)
    )


@pytest.mark.parametrize("length", [63, 64, 65, 255, 256, 257])
def test_ratio_multiword_and_simd_tail_parity(length):
    left = ("abcde" * ((length + 4) // 5))[:length]
    right = ("abxde" * ((length + 4) // 5))[:length]
    assert fuzz.ratio(left, right) == pytest.approx(reference.ratio(left, right))
