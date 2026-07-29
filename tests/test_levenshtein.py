import random

import pytest
from rapidfuzz.distance import Levenshtein as reference

from mojo_rapidfuzz.distance import Levenshtein


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("", ""),
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("Saturday", "Sunday"),
        ("Straße", "strasse"),
        ("你好世界", "你好"),
        (b"\x00\xffabc", b"\x00\xfeabc"),
        ([1, 2, 3], [1, 4, 3, 5]),
    ],
)
def test_distance_parity(left, right):
    assert Levenshtein.distance(left, right) == reference.distance(left, right)


@pytest.mark.parametrize(
    "weights", [(1, 1, 1), (1, 1, 2), (2, 3, 4), (0, 1, 1), (1, 0, 1)]
)
def test_weighted_metrics_parity(weights):
    pairs = [("abc", "xyzq"), ("abc", "a"), ("", "abc"), ("banana", "bandana")]
    for left, right in pairs:
        assert Levenshtein.distance(left, right, weights=weights) == reference.distance(
            left, right, weights=weights
        )
        assert Levenshtein.similarity(
            left, right, weights=weights
        ) == reference.similarity(left, right, weights=weights)
        assert Levenshtein.normalized_distance(
            left, right, weights=weights
        ) == pytest.approx(reference.normalized_distance(left, right, weights=weights))
        assert Levenshtein.normalized_similarity(
            left, right, weights=weights
        ) == pytest.approx(
            reference.normalized_similarity(left, right, weights=weights)
        )


def test_weight_validation_does_not_silently_narrow():
    with pytest.raises(TypeError):
        Levenshtein.distance("a", "b", weights=("2", 1, 1))
    with pytest.raises(OverflowError):
        Levenshtein.distance("a", "b", weights=(2**63, 1, 1))
    assert Levenshtein.distance("a", "b", weights=None) == reference.distance(
        "a", "b", weights=None
    )


def test_processor_and_cutoffs_match():
    processor = str.lower
    assert Levenshtein.distance(
        "MOJO", "mojo", processor=processor
    ) == reference.distance("MOJO", "mojo", processor=processor)
    assert Levenshtein.distance("abc", "xyz", score_cutoff=1) == 2
    assert Levenshtein.similarity("abc", "xyz", score_cutoff=1) == 0
    assert Levenshtein.normalized_distance(
        "abc", "xyz", score_cutoff=0.5
    ) == 1.0


def test_random_distance_parity():
    rng = random.Random(7)
    for _ in range(200):
        left = "".join(rng.choices("abcd", k=rng.randrange(25)))
        right = "".join(rng.choices("abcd", k=rng.randrange(25)))
        assert Levenshtein.distance(left, right) == reference.distance(left, right)


@pytest.mark.parametrize("length", [63, 64, 65, 255, 256, 257])
def test_ascii_multiword_and_simd_tail_parity(length):
    left = ("abcd" * ((length + 3) // 4))[:length]
    right = ("abxd" * ((length + 3) // 4))[:length]
    assert Levenshtein.distance(left, right) == reference.distance(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("spam", "park"),
        ("ab", "ba"),
        ("kitten", "sitting"),
        ("", "abc"),
        ("same", "same"),
    ],
)
def test_editops_are_optimal_and_apply(left, right):
    operations = Levenshtein.editops(left, right)
    assert len(operations) == len(reference.editops(left, right))
    assert operations.apply(left, right) == right
    assert operations.inverse().apply(right, left) == left
    assert operations.as_opcodes().apply(left, right) == right


def test_published_editops_vector_matches():
    assert Levenshtein.editops("spam", "park").as_list() == reference.editops(
        "spam", "park"
    ).as_list()
    assert Levenshtein.opcodes("kitten", "sitting").as_list() == reference.opcodes(
        "kitten", "sitting"
    ).as_list()
