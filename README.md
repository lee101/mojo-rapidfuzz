# mojo-rapidfuzz

`mojo-rapidfuzz` is a standalone Mojo implementation of Levenshtein distance
and fuzzy string matching, with a Python API shaped like
[RapidFuzz](https://github.com/rapidfuzz/RapidFuzz). It is intended for Mojo
projects that need these algorithms without a C++ dependency, and for Python
code that wants to experiment with the same kernels through a small ctypes
bridge.

```python
from mojo_rapidfuzz import fuzz, process
from mojo_rapidfuzz.distance import Levenshtein

assert Levenshtein.distance("kitten", "sitting") == 3
assert fuzz.ratio("this is a test", "this is a test!") > 96

match = process.extractOne(
    "new york jets",
    ["Atlanta Falcons", "New York Jets", "New York Giants"],
)
print(match)
```

The covered functions keep RapidFuzz's names, keyword-only arguments, cutoff
semantics, processor hooks, return scales, and sequence behavior for the cases
exercised by the parity suite. Porting code within this subset generally means
changing `rapidfuzz` to `mojo_rapidfuzz` in the import.

## Coverage

| module | implemented |
| --- | --- |
| `distance.Levenshtein` | `distance`, `similarity`, `normalized_distance`, `normalized_similarity`, `editops`, `opcodes`; non-negative insertion/deletion/substitution weights representable by the Mojo ABI |
| `fuzz` | `ratio`, `partial_ratio`, `partial_ratio_alignment`, `token_sort_ratio`, `token_set_ratio`, `token_ratio`, all three partial token scorers, `WRatio`, `QRatio` |
| `process` | `extractOne`, `extract`, `extract_iter`, `cdist`, `cpdist`; sequence and mapping choices |
| `utils` | `default_process` |

Strings, bytes, and generic sequences of hashable values are supported by the
distance and base ratio scorers. Unicode strings cross the FFI as integer code
points, so matching is by Python character rather than UTF-8 byte.

This is deliberately not all of RapidFuzz. Damerau-Levenshtein, Hamming, Indel
as a standalone distance module, Jaro/Jaro-Winkler, LCSseq, OSA, Prefix,
Postfix, cached scorer classes, and the C-API capsules are not implemented.
`workers` is accepted by `cdist` and `cpdist` for signature compatibility but
does not control execution. ASCII ratio matrices may use CPU parallelism
automatically. Edit scripts are always minimal and produce the same transformed
value; when several minimal scripts exist, their tie selection can differ from
RapidFuzz's bit-parallel traceback.

## Install and run

This repository currently supports installation from a source checkout with
Pixi. Pixi supplies the pinned Mojo nightly, Python, NumPy, pytest, and upstream
RapidFuzz used by the parity suite:

```bash
pixi install
pixi run build
pixi run python
```

Paste the usage example above at the Python prompt. For non-interactive use,
save it as `example.py` and run `pixi run python example.py`. The build task
creates the shared library under `dist/`. Within a source checkout the Python
bridge rebuilds a missing or stale library on first use. A deployment with a
prebuilt library can point `MOJO_RAPIDFUZZ_LIB` at it and does not need a Mojo
compiler at runtime. Wheels and conda packages are not currently published.

## Performance

Measured by an actual `pixi run bench` during the final review on the
machine reported by that command: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64. These are median wall times from the checked-in benchmark, using the
same inputs for both libraries.

| case | mojo-rapidfuzz | rapidfuzz 3.14.5 | result |
| --- | ---: | ---: | ---: |
| Levenshtein.distance (256 chars) | 0.043 ms | 0.009 ms | 0.20x, slower |
| fuzz.ratio (256 chars) | 0.016 ms | 0.003 ms | 0.19x, slower |
| fuzz.partial_ratio (40 vs 87) | 0.057 ms | 0.007 ms | 0.13x, slower |
| process.extractOne ratio (10k) | 4.774 ms | 1.772 ms | 0.37x, slower |
| process.cdist ratio (100x200) | 1.225 ms | 0.376 ms | 0.31x, slower |

Upstream still wins every standard case. The optimized port now uses multiword
bit-parallel kernels for arbitrary-length ASCII unit-cost distance and ratio,
but RapidFuzz's mature C++ dispatch and Python integration remain faster.

`process.cdist(..., scorer=fuzz.ratio, device="gpu")` requests the optional GPU
path; CPU remains the default. Unsupported inputs, unavailable hardware,
insufficient device memory, or a device failure fall back to CPU. No GPU
performance claim is made by the default benchmark.

## How it works

All computation is compiled from `src/rapidfuzz.mojo` into one shared library.
ASCII strings use cached byte views and bit masks. Unicode is encoded to fixed-
width code points so comparisons follow Python characters rather than encoded
bytes. Generic sequences are mapped to contiguous integer symbol IDs. NumPy
owns every contiguous buffer for the full duration of each synchronous ctypes
call; the bridge validates dtype, alignment, layout, and non-null addresses
before passing them across the C ABI.

Unit-cost ASCII Levenshtein and Indel/LCS scoring use multiword bitsets. SIMD
handles complete vectors and a scalar tail handles the remainder. Weighted and
general-sequence scoring uses a one-row dynamic program with common-prefix and
suffix trimming. Batch operations flatten each collection once, reuse query
masks, and can parallelize large matrices or explicitly request the optional
GPU kernel.

## License

MIT
