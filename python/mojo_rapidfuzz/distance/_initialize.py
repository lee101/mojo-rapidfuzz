from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple


class Editop(NamedTuple):
    tag: str
    src_pos: int
    dest_pos: int


class Opcode(NamedTuple):
    tag: str
    src_start: int
    src_end: int
    dest_start: int
    dest_end: int


class MatchingBlock(NamedTuple):
    a: int
    b: int
    size: int


class _ListLike(Sequence):
    def __init__(self, values: Iterable, src_len: int, dest_len: int):
        self._values = list(values)
        self.src_len = src_len
        self.dest_len = dest_len

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, item):
        return self._values[item]

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def as_list(self):
        return [tuple(value) for value in self._values]


class Editops(_ListLike):
    def __init__(self, values=(), src_len=0, dest_len=0):
        super().__init__((Editop(*value) for value in values), src_len, dest_len)

    def __repr__(self):
        return (
            f"Editops({self._values!r}, src_len={self.src_len}, "
            f"dest_len={self.dest_len})"
        )

    def inverse(self):
        tags = {"insert": "delete", "delete": "insert", "replace": "replace"}
        return Editops(
            [
                Editop(tags[op.tag], op.dest_pos, op.src_pos)
                for op in self._values
            ],
            self.dest_len,
            self.src_len,
        )

    def apply(self, source_string, destination_string):
        result = source_string[:0]
        cursor = 0
        for op in self._values:
            result += source_string[cursor : op.src_pos]
            if op.tag in ("insert", "replace"):
                result += destination_string[op.dest_pos : op.dest_pos + 1]
            cursor = op.src_pos + (op.tag != "insert")
        return result + source_string[cursor:]

    def as_opcodes(self):
        return _opcodes_from_editops(self)

    def as_matching_blocks(self):
        return self.as_opcodes().as_matching_blocks()

    def remove_subsequence(self, subsequence):
        remaining = list(self._values)
        for wanted in subsequence:
            wanted = Editop(*wanted)
            try:
                remaining.remove(wanted)
            except ValueError as exc:
                raise ValueError("subsequence is not a subsequence") from exc
        return Editops(remaining, self.src_len, self.dest_len)


class Opcodes(_ListLike):
    def __init__(self, values=(), src_len=0, dest_len=0):
        super().__init__((Opcode(*value) for value in values), src_len, dest_len)

    def __repr__(self):
        return (
            f"Opcodes({self._values!r}, src_len={self.src_len}, "
            f"dest_len={self.dest_len})"
        )

    def inverse(self):
        tags = {
            "insert": "delete",
            "delete": "insert",
            "replace": "replace",
            "equal": "equal",
        }
        return Opcodes(
            [
                Opcode(
                    tags[op.tag],
                    op.dest_start,
                    op.dest_end,
                    op.src_start,
                    op.src_end,
                )
                for op in self._values
            ],
            self.dest_len,
            self.src_len,
        )

    def apply(self, source_string, destination_string):
        result = source_string[:0]
        for op in self._values:
            if op.tag == "equal":
                result += source_string[op.src_start : op.src_end]
            elif op.tag in ("insert", "replace"):
                result += destination_string[op.dest_start : op.dest_end]
        return result

    def as_editops(self):
        values = []
        for op in self._values:
            if op.tag == "equal":
                continue
            src = op.src_start
            dest = op.dest_start
            while src < op.src_end or dest < op.dest_end:
                if src < op.src_end and dest < op.dest_end:
                    values.append(Editop("replace", src, dest))
                    src += 1
                    dest += 1
                elif src < op.src_end:
                    values.append(Editop("delete", src, dest))
                    src += 1
                else:
                    values.append(Editop("insert", src, dest))
                    dest += 1
        return Editops(values, self.src_len, self.dest_len)

    def as_matching_blocks(self):
        blocks = [
            MatchingBlock(
                op.src_start, op.dest_start, op.src_end - op.src_start
            )
            for op in self._values
            if op.tag == "equal"
        ]
        blocks.append(MatchingBlock(self.src_len, self.dest_len, 0))
        return blocks


def _opcodes_from_editops(editops: Editops) -> Opcodes:
    elementary = []
    src = dest = 0
    for op in editops:
        if src < op.src_pos or dest < op.dest_pos:
            elementary.append(Opcode("equal", src, op.src_pos, dest, op.dest_pos))
        if op.tag == "replace":
            elementary.append(
                Opcode("replace", op.src_pos, op.src_pos + 1, op.dest_pos, op.dest_pos + 1)
            )
            src, dest = op.src_pos + 1, op.dest_pos + 1
        elif op.tag == "delete":
            elementary.append(
                Opcode("delete", op.src_pos, op.src_pos + 1, op.dest_pos, op.dest_pos)
            )
            src, dest = op.src_pos + 1, op.dest_pos
        else:
            elementary.append(
                Opcode("insert", op.src_pos, op.src_pos, op.dest_pos, op.dest_pos + 1)
            )
            src, dest = op.src_pos, op.dest_pos + 1
    if src < editops.src_len or dest < editops.dest_len:
        elementary.append(
            Opcode("equal", src, editops.src_len, dest, editops.dest_len)
        )

    merged = []
    for op in elementary:
        if (
            merged
            and merged[-1].tag == op.tag
            and merged[-1].src_end == op.src_start
            and merged[-1].dest_end == op.dest_start
        ):
            previous = merged[-1]
            merged[-1] = Opcode(
                op.tag,
                previous.src_start,
                op.src_end,
                previous.dest_start,
                op.dest_end,
            )
        else:
            merged.append(op)
    return Opcodes(merged, editops.src_len, editops.dest_len)
