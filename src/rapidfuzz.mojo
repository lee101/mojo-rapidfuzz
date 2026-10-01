"""Levenshtein and Indel kernels used by the Python bindings."""

from std.bit import pop_count
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def imin(a: Int, b: Int) -> Int:
    return a if a < b else b


def imin3(a: Int, b: Int, c: Int) -> Int:
    return imin(imin(a, b), c)


def popcount(value: UInt64) -> Int:
    var bits = value
    var count = 0
    while bits != 0:
        bits &= bits - 1
        count += 1
    return count


def zero_words(pointer: U64Ptr, count: Int):
    comptime W = simd_width_of[DType.uint64]()
    var zero = SIMD[DType.uint64, W](0)
    var i = 0
    while i + W <= count:
        pointer.store(i, zero)
        i += W
    while i < count:
        pointer[i] = 0
        i += 1


def popcount_words(pointer: U64Ptr, count: Int) -> Int:
    comptime W = simd_width_of[DType.uint64]()
    var total = 0
    var i = 0
    while i + W <= count:
        total += Int(pop_count(pointer.unsafe_load[width=W](i)).reduce_add())
        i += W
    while i < count:
        total += popcount(pointer[i])
        i += 1
    return total


def build_ascii_masks(pattern: U8Ptr, length: Int, masks: U64Ptr, words: Int):
    zero_words(masks, 256 * words)
    for i in range(length):
        var word = i // 64
        masks[Int(pattern[i]) * words + word] |= UInt64(1) << UInt64(i % 64)


def lcs_ascii_masked(
    text: U8Ptr,
    text_len: Int,
    masks: U64Ptr,
    words: Int,
    state: U64Ptr,
) -> Int:
    zero_words(state, words)
    for i in range(text_len):
        var shift_carry = UInt64(1)
        var borrow = UInt64(0)
        for word in range(words):
            var previous = state[word]
            var combined = masks[Int(text[i]) * words + word] | previous
            var shifted = (previous << 1) | shift_carry
            shift_carry = previous >> 63
            var subtrahend = shifted + borrow
            var overflow = subtrahend < shifted
            var next_borrow = UInt64(
                combined < subtrahend or overflow
            )
            state[word] = combined & ~(combined - subtrahend)
            borrow = next_borrow
    return popcount_words(state, words)


def lcs_ascii(
    pattern: U8Ptr,
    text: U8Ptr,
    pattern_len: Int,
    text_len: Int,
    scratch: U64Ptr,
) -> Int:
    if pattern_len == 0:
        return 0
    var words = (pattern_len + 63) // 64
    build_ascii_masks(pattern, pattern_len, scratch, words)
    return lcs_ascii_masked(
        text, text_len, scratch, words, scratch + 256 * words
    )


def unit_distance_ascii_masked(
    text: U8Ptr,
    pattern_len: Int,
    text_len: Int,
    masks: U64Ptr,
    state: U64Ptr,
) -> Int:
    if pattern_len == 0:
        return text_len
    var words = (pattern_len + 63) // 64
    var positive = state
    var negative = positive + words
    zero_words(negative, words)
    for word in range(words):
        positive[word] = ~UInt64(0)

    var distance = pattern_len
    var final_word = words - 1
    var high_bit = UInt64(1) << UInt64((pattern_len - 1) % 64)
    for i in range(text_len):
        var add_carry = UInt64(0)
        var positive_carry = UInt64(1)
        var negative_carry = UInt64(0)
        for word in range(words):
            var pv = positive[word]
            var nv = negative[word]
            var matches = masks[Int(text[i]) * words + word]
            var x = matches | nv
            var left = x & pv
            var first_sum = left + pv
            var first_carry = first_sum < left
            var full_sum = first_sum + add_carry
            var second_carry = full_sum < first_sum
            add_carry = UInt64(first_carry or second_carry)
            var difference = (full_sum ^ pv) | x
            var horizontal_positive = nv | ~(difference | pv)
            var horizontal_negative = difference & pv
            if word == final_word:
                if (horizontal_positive & high_bit) != 0:
                    distance += 1
                elif (horizontal_negative & high_bit) != 0:
                    distance -= 1
            var next_positive_carry = horizontal_positive >> 63
            var next_negative_carry = horizontal_negative >> 63
            horizontal_positive = (horizontal_positive << 1) | positive_carry
            horizontal_negative = (
                horizontal_negative << 1
            ) | negative_carry
            positive_carry = next_positive_carry
            negative_carry = next_negative_carry
            positive[word] = horizontal_negative | ~(
                difference | horizontal_positive
            )
            negative[word] = horizontal_positive & difference
    return distance


def unit_distance_ascii(
    pattern: U8Ptr,
    text: U8Ptr,
    pattern_len: Int,
    text_len: Int,
    scratch: U64Ptr,
) -> Int:
    if pattern_len == 0:
        return text_len
    var words = (pattern_len + 63) // 64
    build_ascii_masks(pattern, pattern_len, scratch, words)
    return unit_distance_ascii_masked(
        text,
        pattern_len,
        text_len,
        scratch,
        scratch + 256 * words,
    )


def unit_distance_bit(
    pattern: U32Ptr, text: U32Ptr, pattern_len: Int, text_len: Int
) -> Int:
    if pattern_len == 0:
        return text_len
    var positive = ~UInt64(0)
    var negative = UInt64(0)
    var distance = pattern_len
    var high_bit = UInt64(1) << UInt64(pattern_len - 1)
    for i in range(text_len):
        var matches = UInt64(0)
        for j in range(pattern_len):
            if pattern[j] == text[i]:
                matches |= UInt64(1) << UInt64(j)
        var x = matches | negative
        var difference = (((x & positive) + positive) ^ positive) | x
        var horizontal_positive = negative | ~(difference | positive)
        var horizontal_negative = difference & positive
        if (horizontal_positive & high_bit) != 0:
            distance += 1
        elif (horizontal_negative & high_bit) != 0:
            distance -= 1
        horizontal_positive = (horizontal_positive << 1) | UInt64(1)
        horizontal_negative <<= 1
        positive = horizontal_negative | ~(difference | horizontal_positive)
        negative = horizontal_positive & difference
    return distance


def lcs_bit(pattern: U32Ptr, text: U32Ptr, pattern_len: Int, text_len: Int) -> Int:
    var state = UInt64(0)
    for i in range(text_len):
        var matches = UInt64(0)
        for j in range(pattern_len):
            if pattern[j] == text[i]:
                matches |= UInt64(1) << UInt64(j)
        var combined = matches | state
        state = combined & ~(combined - ((state << 1) | UInt64(1)))
    return popcount(state)


def lcs_masked(text: U32Ptr, text_len: Int, masks: U64Ptr) -> Int:
    var state = UInt64(0)
    for i in range(text_len):
        var combined = masks[Int(text[i])] | state
        state = combined & ~(combined - ((state << 1) | UInt64(1)))
    return popcount(state)


def lcs_masked_ascii(text: U8Ptr, text_len: Int, masks: U64Ptr) -> Int:
    var state = UInt64(0)
    for i in range(text_len):
        var combined = masks[Int(text[i])] | state
        state = combined & ~(combined - ((state << 1) | UInt64(1)))
    return popcount(state)


def build_ascii_mask_64(query: U8Ptr, query_len: Int, masks: U64Ptr):
    zero_words(masks, 256)
    for i in range(query_len):
        masks[Int(query[i])] |= UInt64(1) << UInt64(i)


def weighted_distance(
    a: U32Ptr,
    b: U32Ptr,
    n: Int,
    m: Int,
    insert_cost: Int,
    delete_cost: Int,
    replace_cost: Int,
    work: I64Ptr,
) -> Int:
    var prefix = 0
    var nn = n
    var mm = m
    while prefix < nn and prefix < mm and a[prefix] == b[prefix]:
        prefix += 1
    while nn > prefix and mm > prefix and a[nn - 1] == b[mm - 1]:
        nn -= 1
        mm -= 1

    var an = nn - prefix
    var bm = mm - prefix
    if an == 0:
        return bm * insert_cost
    if bm == 0:
        return an * delete_cost
    if (
        insert_cost == 1
        and delete_cost == 1
        and replace_cost == 1
        and (an <= 63 or bm <= 63)
    ):
        if an <= bm:
            return unit_distance_bit(a + prefix, b + prefix, an, bm)
        return unit_distance_bit(b + prefix, a + prefix, bm, an)

    for j in range(bm + 1):
        work[j] = Int64(j * insert_cost)
    for i in range(1, an + 1):
        var diagonal = Int(work[0])
        work[0] = Int64(i * delete_cost)
        for j in range(1, bm + 1):
            var above = Int(work[j])
            var substitution = 0 if a[prefix + i - 1] == b[prefix + j - 1] else replace_cost
            work[j] = Int64(
                imin3(
                    above + delete_cost,
                    Int(work[j - 1]) + insert_cost,
                    diagonal + substitution,
                )
            )
            diagonal = above
    return Int(work[bm])


def lcs_length(a: U32Ptr, b: U32Ptr, n: Int, m: Int, work: I64Ptr) -> Int:
    var prefix = 0
    var nn = n
    var mm = m
    while prefix < nn and prefix < mm and a[prefix] == b[prefix]:
        prefix += 1
    while nn > prefix and mm > prefix and a[nn - 1] == b[mm - 1]:
        nn -= 1
        mm -= 1

    var an = nn - prefix
    var bm = mm - prefix
    if an == 0 or bm == 0:
        return prefix + (n - nn)
    if an <= 63:
        return prefix + (n - nn) + lcs_bit(a + prefix, b + prefix, an, bm)
    if bm <= 63:
        return prefix + (n - nn) + lcs_bit(b + prefix, a + prefix, bm, an)

    for j in range(bm + 1):
        work[j] = 0
    for i in range(1, an + 1):
        var diagonal = Int64(0)
        for j in range(1, bm + 1):
            var above = work[j]
            if a[prefix + i - 1] == b[prefix + j - 1]:
                work[j] = diagonal + 1
            elif work[j - 1] > above:
                work[j] = work[j - 1]
            diagonal = above
    return prefix + (n - nn) + Int(work[bm])


def ratio_score(a: U32Ptr, b: U32Ptr, n: Int, m: Int, work: I64Ptr) -> Float64:
    var total = n + m
    if total == 0:
        return 100.0
    var lcs = lcs_length(a, b, n, m, work)
    return 100.0 * Float64(2 * lcs) / Float64(total)


def partial_ratio_one_way(
    needle: U32Ptr,
    haystack: U32Ptr,
    n: Int,
    m: Int,
    work: I64Ptr,
    alignment: I64Ptr,
) -> Float64:
    var best = 0.0
    var best_start = 0
    var best_end = n

    if m >= n:
        for start in range(m - n + 1):
            var score = ratio_score(needle, haystack + start, n, n, work)
            if score > best:
                best = score
                best_start = start
                best_end = start + n
                if best == 100.0:
                    alignment[0] = Int64(best_start)
                    alignment[1] = Int64(best_end)
                    return best

    for length in range(1, n):
        var score = ratio_score(needle, haystack, n, length, work)
        if score > best:
            best = score
            best_start = 0
            best_end = length
            if best == 100.0:
                alignment[0] = Int64(best_start)
                alignment[1] = Int64(best_end)
                return best

    var first = m - n
    if first < 0:
        first = 0
    for start in range(first, m):
        var score = ratio_score(needle, haystack + start, n, m - start, work)
        if score > best:
            best = score
            best_start = start
            best_end = m
            if best == 100.0:
                alignment[0] = Int64(best_start)
                alignment[1] = Int64(best_end)
                return best

    alignment[0] = Int64(best_start)
    alignment[1] = Int64(best_end)
    return best


def partial_ratio_one_way_ascii(
    needle: U8Ptr,
    haystack: U8Ptr,
    n: Int,
    m: Int,
    scratch: U64Ptr,
    alignment: I64Ptr,
) -> Float64:
    var masks = scratch
    var state = scratch + 256
    build_ascii_masks(needle, n, masks, 1)
    var best = 0.0
    var best_start = 0
    var best_end = n

    if m >= n:
        for start in range(m - n + 1):
            var lcs = lcs_ascii_masked(haystack + start, n, masks, 1, state)
            var score = 100.0 * Float64(lcs) / Float64(n)
            if score > best:
                best = score
                best_start = start
                best_end = start + n
                if best == 100.0:
                    alignment[0] = Int64(best_start)
                    alignment[1] = Int64(best_end)
                    return best

    for length in range(1, n):
        var lcs = lcs_ascii_masked(haystack, length, masks, 1, state)
        var score = 200.0 * Float64(lcs) / Float64(n + length)
        if score > best:
            best = score
            best_start = 0
            best_end = length

    var first = m - n
    if first < 0:
        first = 0
    for start in range(first, m):
        var length = m - start
        var lcs = lcs_ascii_masked(haystack + start, length, masks, 1, state)
        var score = 200.0 * Float64(lcs) / Float64(n + length)
        if score > best:
            best = score
            best_start = start
            best_end = m

    alignment[0] = Int64(best_start)
    alignment[1] = Int64(best_end)
    return best


@export("mrf_levenshtein_distance")
def mrf_levenshtein_distance(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    m: Int,
    insert_cost: Int,
    delete_cost: Int,
    replace_cost: Int,
    work_addr: Int,
) abi("C") -> Int:
    return weighted_distance(
        U32Ptr(unsafe_from_address=a_addr),
        U32Ptr(unsafe_from_address=b_addr),
        n,
        m,
        insert_cost,
        delete_cost,
        replace_cost,
        I64Ptr(unsafe_from_address=work_addr),
    )


@export("mrf_levenshtein_distance_ascii")
def mrf_levenshtein_distance_ascii(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    m: Int,
    scratch_addr: Int,
) abi("C") -> Int:
    var a = U8Ptr(unsafe_from_address=a_addr)
    var b = U8Ptr(unsafe_from_address=b_addr)
    var scratch = U64Ptr(unsafe_from_address=scratch_addr)
    if n <= m:
        return unit_distance_ascii(a, b, n, m, scratch)
    return unit_distance_ascii(b, a, m, n, scratch)


@export("mrf_levenshtein_distance_ascii_masked")
def mrf_levenshtein_distance_ascii_masked(
    text_addr: Int,
    pattern_len: Int,
    text_len: Int,
    masks_addr: Int,
    state_addr: Int,
) abi("C") -> Int:
    return unit_distance_ascii_masked(
        U8Ptr(unsafe_from_address=text_addr),
        pattern_len,
        text_len,
        U64Ptr(unsafe_from_address=masks_addr),
        U64Ptr(unsafe_from_address=state_addr),
    )


@export("mrf_ratio")
def mrf_ratio(
    a_addr: Int, b_addr: Int, n: Int, m: Int, work_addr: Int
) abi("C") -> Float64:
    return ratio_score(
        U32Ptr(unsafe_from_address=a_addr),
        U32Ptr(unsafe_from_address=b_addr),
        n,
        m,
        I64Ptr(unsafe_from_address=work_addr),
    )


@export("mrf_ratio_ascii")
def mrf_ratio_ascii(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    m: Int,
    scratch_addr: Int,
) abi("C") -> Float64:
    var total = n + m
    if total == 0:
        return 100.0
    var a = U8Ptr(unsafe_from_address=a_addr)
    var b = U8Ptr(unsafe_from_address=b_addr)
    var scratch = U64Ptr(unsafe_from_address=scratch_addr)
    var lcs = (
        lcs_ascii(a, b, n, m, scratch)
        if n <= m
        else lcs_ascii(b, a, m, n, scratch)
    )
    return 200.0 * Float64(lcs) / Float64(total)


@export("mrf_ratio_ascii_masked")
def mrf_ratio_ascii_masked(
    text_addr: Int,
    pattern_len: Int,
    text_len: Int,
    masks_addr: Int,
    state_addr: Int,
) abi("C") -> Float64:
    var total = pattern_len + text_len
    if total == 0:
        return 100.0
    var words = (pattern_len + 63) // 64
    var lcs = lcs_ascii_masked(
        U8Ptr(unsafe_from_address=text_addr),
        text_len,
        U64Ptr(unsafe_from_address=masks_addr),
        words,
        U64Ptr(unsafe_from_address=state_addr),
    )
    return 200.0 * Float64(lcs) / Float64(total)


@export("mrf_partial_ratio")
def mrf_partial_ratio(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    m: Int,
    work_addr: Int,
    alignment_addr: Int,
) abi("C") -> Float64:
    var a = U32Ptr(unsafe_from_address=a_addr)
    var b = U32Ptr(unsafe_from_address=b_addr)
    var work = I64Ptr(unsafe_from_address=work_addr)
    var alignment = I64Ptr(unsafe_from_address=alignment_addr)
    if n == 0 or m == 0:
        alignment[0] = 0
        alignment[1] = Int64(n)
        alignment[2] = 0
        alignment[3] = Int64(m)
        return 100.0 if n == m else 0.0
    if n < m:
        var score = partial_ratio_one_way(a, b, n, m, work, alignment + 2)
        alignment[0] = 0
        alignment[1] = Int64(n)
        return score
    if m < n:
        var score = partial_ratio_one_way(b, a, m, n, work, alignment)
        alignment[2] = 0
        alignment[3] = Int64(m)
        return score

    var score = partial_ratio_one_way(a, b, n, m, work, alignment + 2)
    var first_dest_start = alignment[2]
    var first_dest_end = alignment[3]
    alignment[0] = 0
    alignment[1] = Int64(n)
    if score < 100.0:
        var other_alignment = alignment
        var other_score = partial_ratio_one_way(b, a, m, n, work, other_alignment)
        if other_score > score:
            alignment[2] = 0
            alignment[3] = Int64(m)
            score = other_score
        else:
            alignment[0] = 0
            alignment[1] = Int64(n)
            alignment[2] = first_dest_start
            alignment[3] = first_dest_end
    return score


@export("mrf_partial_ratio_ascii")
def mrf_partial_ratio_ascii(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    m: Int,
    scratch_addr: Int,
    alignment_addr: Int,
) abi("C") -> Float64:
    var a = U8Ptr(unsafe_from_address=a_addr)
    var b = U8Ptr(unsafe_from_address=b_addr)
    var scratch = U64Ptr(unsafe_from_address=scratch_addr)
    var alignment = I64Ptr(unsafe_from_address=alignment_addr)
    if n == 0 or m == 0:
        alignment[0] = 0
        alignment[1] = Int64(n)
        alignment[2] = 0
        alignment[3] = Int64(m)
        return 100.0 if n == m else 0.0
    if n < m:
        var score = partial_ratio_one_way_ascii(
            a, b, n, m, scratch, alignment + 2
        )
        alignment[0] = 0
        alignment[1] = Int64(n)
        return score
    if m < n:
        var score = partial_ratio_one_way_ascii(
            b, a, m, n, scratch, alignment
        )
        alignment[2] = 0
        alignment[3] = Int64(m)
        return score

    var score = partial_ratio_one_way_ascii(
        a, b, n, m, scratch, alignment + 2
    )
    var first_dest_start = alignment[2]
    var first_dest_end = alignment[3]
    alignment[0] = 0
    alignment[1] = Int64(n)
    if score < 100.0:
        var other_score = partial_ratio_one_way_ascii(
            b, a, m, n, scratch, alignment
        )
        if other_score > score:
            alignment[2] = 0
            alignment[3] = Int64(m)
            score = other_score
        else:
            alignment[0] = 0
            alignment[1] = Int64(n)
            alignment[2] = first_dest_start
            alignment[3] = first_dest_end
    return score


@export("mrf_ratio_many")
def mrf_ratio_many(
    query_addr: Int,
    query_len: Int,
    choices_addr: Int,
    offsets_addr: Int,
    count: Int,
    work_addr: Int,
    scores_addr: Int,
    masks_addr: Int,
) abi("C"):
    var query = U32Ptr(unsafe_from_address=query_addr)
    var choices = U32Ptr(unsafe_from_address=choices_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var work = I64Ptr(unsafe_from_address=work_addr)
    var scores = F64Ptr(unsafe_from_address=scores_addr)
    var masks = U64Ptr(unsafe_from_address=masks_addr)
    for i in range(count):
        var start = Int(offsets[i])
        var length = Int(offsets[i + 1]) - start
        if query_len <= 63:
            var total = query_len + length
            scores[i] = (
                100.0
                if total == 0
                else 100.0
                * Float64(2 * lcs_masked(choices + start, length, masks))
                / Float64(total)
            )
        else:
            scores[i] = ratio_score(query, choices + start, query_len, length, work)


@export("mrf_ratio_many_ascii")
def mrf_ratio_many_ascii(
    query_addr: Int,
    query_len: Int,
    choices_addr: Int,
    offsets_addr: Int,
    count: Int,
    scores_addr: Int,
    masks_addr: Int,
) abi("C"):
    var query = U8Ptr(unsafe_from_address=query_addr)
    var choices = U8Ptr(unsafe_from_address=choices_addr)
    var offsets = I64Ptr(unsafe_from_address=offsets_addr)
    var scores = F64Ptr(unsafe_from_address=scores_addr)
    var masks = U64Ptr(unsafe_from_address=masks_addr)
    build_ascii_mask_64(query, query_len, masks)
    for i in range(count):
        var start = Int(offsets[i])
        var length = Int(offsets[i + 1]) - start
        var total = query_len + length
        scores[i] = (
            100.0
            if total == 0
            else 200.0
            * Float64(lcs_masked_ascii(choices + start, length, masks))
            / Float64(total)
        )


@export("mrf_ratio_matrix_ascii")
def mrf_ratio_matrix_ascii(
    queries_addr: Int,
    query_offsets_addr: Int,
    query_count: Int,
    choices_addr: Int,
    choice_offsets_addr: Int,
    choice_count: Int,
    scores_addr: Int,
    masks_addr: Int,
) abi("C"):
    var queries = U8Ptr(unsafe_from_address=queries_addr)
    var query_offsets = I64Ptr(unsafe_from_address=query_offsets_addr)
    var choices = U8Ptr(unsafe_from_address=choices_addr)
    var choice_offsets = I64Ptr(unsafe_from_address=choice_offsets_addr)
    var scores = F64Ptr(unsafe_from_address=scores_addr)
    var all_masks = U64Ptr(unsafe_from_address=masks_addr)

    @always_inline
    def score_row(row: Int) capturing:
        var query_start = Int(query_offsets[row])
        var query_len = Int(query_offsets[row + 1]) - query_start
        var masks = all_masks + row * 256
        build_ascii_mask_64(queries + query_start, query_len, masks)
        for column in range(choice_count):
            var choice_start = Int(choice_offsets[column])
            var choice_len = (
                Int(choice_offsets[column + 1]) - choice_start
            )
            var total = query_len + choice_len
            scores[row * choice_count + column] = (
                100.0
                if total == 0
                else 200.0
                * Float64(
                    lcs_masked_ascii(
                        choices + choice_start, choice_len, masks
                    )
                )
                / Float64(total)
            )

    for row in range(query_count):
        score_row(row)


def prepare_ascii_masks_gpu(
    queries: U8Ptr,
    query_offsets: I64Ptr,
    query_count: Int64,
    masks: U64Ptr,
):
    # Device-passable scalars must be fixed-width; 1.2.0 rejects `Int` here.
    var row = Int(global_idx.x)
    if row >= Int(query_count):
        return
    var row_masks = masks + row * 256
    for i in range(256):
        row_masks[i] = 0
    var start = Int(query_offsets[row])
    var length = Int(query_offsets[row + 1]) - start
    for i in range(length):
        row_masks[Int(queries[start + i])] |= UInt64(1) << UInt64(i)


def ratio_matrix_ascii_gpu_kernel(
    queries: U8Ptr,
    query_offsets: I64Ptr,
    query_count: Int64,
    choices: U8Ptr,
    choice_offsets: I64Ptr,
    choice_count: Int64,
    scores: F64Ptr,
    masks: U64Ptr,
):
    # Device-passable scalars must be fixed-width; 1.2.0 rejects `Int` here.
    var index = Int(global_idx.x)
    var queries_total = Int(query_count)
    var choices_total = Int(choice_count)
    var size = queries_total * choices_total
    if index >= size:
        return
    var row = index // choices_total
    var column = index - row * choices_total
    var query_len = (
        Int(query_offsets[row + 1]) - Int(query_offsets[row])
    )
    var choice_start = Int(choice_offsets[column])
    var choice_len = Int(choice_offsets[column + 1]) - choice_start
    var state = UInt64(0)
    var row_masks = masks + row * 256
    for i in range(choice_len):
        var combined = row_masks[Int(choices[choice_start + i])] | state
        state = combined & ~(combined - ((state << 1) | UInt64(1)))
    var total = query_len + choice_len
    scores[index] = (
        100.0
        if total == 0
        else 200.0 * Float64(popcount(state)) / Float64(total)
    )


@export("mrf_ratio_matrix_ascii_gpu")
def mrf_ratio_matrix_ascii_gpu(
    queries_addr: Int,
    query_size: Int,
    query_offsets_addr: Int,
    query_count: Int,
    choices_addr: Int,
    choice_size: Int,
    choice_offsets_addr: Int,
    choice_count: Int,
    scores_addr: Int,
) abi("C") -> Int:
    try:
        var ctx = DeviceContext()
        var queries_device = ctx.enqueue_create_buffer[DType.uint8](query_size)
        var query_offsets_device = ctx.enqueue_create_buffer[DType.int64](
            query_count + 1
        )
        var choices_device = ctx.enqueue_create_buffer[DType.uint8](choice_size)
        var choice_offsets_device = ctx.enqueue_create_buffer[DType.int64](
            choice_count + 1
        )
        var scores_device = ctx.enqueue_create_buffer[DType.float64](
            query_count * choice_count
        )
        var masks_device = ctx.enqueue_create_buffer[DType.uint64](
            query_count * 256
        )
        ctx.enqueue_copy(
            queries_device, U8Ptr(unsafe_from_address=queries_addr)
        )
        ctx.enqueue_copy(
            query_offsets_device,
            I64Ptr(unsafe_from_address=query_offsets_addr),
        )
        ctx.enqueue_copy(
            choices_device, U8Ptr(unsafe_from_address=choices_addr)
        )
        ctx.enqueue_copy(
            choice_offsets_device,
            I64Ptr(unsafe_from_address=choice_offsets_addr),
        )
        comptime BLOCK_SIZE = 256
        ctx.enqueue_function[prepare_ascii_masks_gpu](
            queries_device,
            query_offsets_device,
            Int64(query_count),
            masks_device,
            grid_dim=(query_count + BLOCK_SIZE - 1) // BLOCK_SIZE,
            block_dim=BLOCK_SIZE,
        )
        var size = query_count * choice_count
        ctx.enqueue_function[ratio_matrix_ascii_gpu_kernel](
            queries_device,
            query_offsets_device,
            Int64(query_count),
            choices_device,
            choice_offsets_device,
            Int64(choice_count),
            scores_device,
            masks_device,
            grid_dim=(size + BLOCK_SIZE - 1) // BLOCK_SIZE,
            block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_copy(
            F64Ptr(unsafe_from_address=scores_addr), scores_device
        )
        ctx.synchronize()
        return 1
    except:
        return 0


@export("mrf_levenshtein_matrix")
def mrf_levenshtein_matrix(
    a_addr: Int, b_addr: Int, n: Int, m: Int, matrix_addr: Int
) abi("C"):
    var a = U32Ptr(unsafe_from_address=a_addr)
    var b = U32Ptr(unsafe_from_address=b_addr)
    var matrix = I64Ptr(unsafe_from_address=matrix_addr)
    var width = m + 1
    for i in range(n + 1):
        matrix[i * width] = Int64(i)
    for j in range(m + 1):
        matrix[j] = Int64(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            var substitution = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i * width + j] = Int64(
                imin3(
                    Int(matrix[(i - 1) * width + j]) + 1,
                    Int(matrix[i * width + j - 1]) + 1,
                    Int(matrix[(i - 1) * width + j - 1]) + substitution,
                )
            )
