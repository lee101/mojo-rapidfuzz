from __future__ import annotations

def default_process(sentence: str) -> str:
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a String")
    return "".join(char if char.isalnum() else " " for char in sentence).strip().lower()
