import numpy as np
import pytest
from rapidfuzz import fuzz as reference_fuzz
from rapidfuzz import process as reference
from rapidfuzz import utils as reference_utils

from mojo_rapidfuzz import fuzz, process, utils
from mojo_rapidfuzz._lib import addr
from mojo_rapidfuzz.distance import Levenshtein


def test_extract_one_parity():
    choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
    assert process.extractOne(
        "new york jets", choices, processor=utils.default_process
    ) == pytest.approx(
        reference.extractOne(
            "new york jets", choices, processor=reference_utils.default_process
        )
    )


def test_extract_mapping_and_ratio_batch_parity():
    choices = {"short": "ab", "exact": "abcd", "near": "axcd"}
    assert process.extract(
        "abcd", choices, scorer=fuzz.ratio, limit=None
    ) == reference.extract("abcd", choices, scorer=reference_fuzz.ratio, limit=None)


def test_extract_distance_scorer_parity():
    choices = ["ab", "abcd", "axcd", "zzzz"]
    assert process.extract(
        "abcd",
        choices,
        scorer=Levenshtein.distance,
        limit=None,
        score_cutoff=2,
    ) == reference.extract(
        "abcd",
        choices,
        scorer=__import__(
            "rapidfuzz.distance", fromlist=["Levenshtein"]
        ).Levenshtein.distance,
        limit=None,
        score_cutoff=2,
    )


def test_extract_iter_skips_none():
    choices = ["abc", None, "xbc"]
    assert list(process.extract_iter("abc", choices, scorer=fuzz.ratio)) == [
        ("abc", 100.0, 0),
        ("xbc", pytest.approx(66.66666666666667), 2),
    ]


def test_scorer_type_error_is_not_retried_or_swallowed():
    calls = 0

    def broken_scorer(left, right, *, score_cutoff=None, score_hint=None):
        nonlocal calls
        calls += 1
        raise TypeError("scorer implementation failed")

    with pytest.raises(TypeError, match="implementation failed"):
        list(
            process.extract_iter(
                "abc", ["abc"], scorer=broken_scorer, score_hint=100
            )
        )
    assert calls == 1


def test_cdist_parity():
    queries = ["abc", "ab", "new york"]
    choices = ["abc", "xbc", "new york city"]
    ours = process.cdist(queries, choices, scorer=fuzz.ratio)
    theirs = reference.cdist(queries, choices, scorer=reference_fuzz.ratio)
    assert ours.dtype == theirs.dtype
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize(("rows", "columns"), [(511, 513), (512, 512)])
def test_cdist_parallel_threshold_parity(rows, columns):
    queries = ["abc"] * rows
    choices = ["axc"] * columns
    ours = process.cdist(queries, choices, scorer=fuzz.ratio)
    assert ours.shape == (rows, columns)
    assert np.all(ours == np.float32(66.66666666666667))


def test_cdist_gpu_path_or_cpu_fallback_parity():
    queries = ["abc", "kitten", ""]
    choices = ["xbc", "sitting", ""]
    cpu = process.cdist(queries, choices, scorer=fuzz.ratio, device="cpu")
    gpu = process.cdist(queries, choices, scorer=fuzz.ratio, device="gpu")
    assert np.array_equal(cpu, gpu)


def test_cdist_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        process.cdist(["abc"], ["abc"], device="tpu")


def test_ffi_address_rejects_wrong_dtype_and_layout():
    with pytest.raises(TypeError, match="dtype"):
        addr(np.ones(1, dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((2, 2), dtype=np.float64)[:, 0])


def test_cpdist_parity():
    queries = ["abc", "ab", "new york"]
    choices = ["abc", "xbc", "new york city"]
    ours = process.cpdist(queries, choices, scorer=fuzz.ratio)
    theirs = reference.cpdist(queries, choices, scorer=reference_fuzz.ratio)
    assert np.array_equal(ours, theirs)


def test_workers_is_accepted_for_signature_compatibility():
    assert process.cdist(["a"], ["a"], workers=-1)[0, 0] == 100
    assert process.cpdist(["a"], ["a"], workers=-1)[0] == 100


def test_default_process_parity():
    for value in ["THIS is a TEST!!!", "  hello_world  ", "Straße", ""]:
        assert utils.default_process(value) == reference_utils.default_process(value)
