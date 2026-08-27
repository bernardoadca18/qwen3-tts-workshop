from __future__ import annotations

import hashlib
import json
import re

from voice_service.domain import TextChunk

_PARAGRAPH = re.compile(r"\n[ \t]*\n+")
_SENTENCE = re.compile(r"[.!?\u2026]+(?:[\"')\]]+)?(?:\s+|$)")
_WHITESPACE = re.compile(r"\s+")
CHUNKING_STRATEGY_VERSION = "qwen3-tts-chunking-v2"


def derive_chunk_seed(base_seed: int, chunk_index: int) -> int:
    payload = f"qwen3-tts-chunk-v1:{base_seed}:{chunk_index}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**31 - 1) + 1


class TextChunker:
    def __init__(self, *, target_chars: int, max_chars: int):
        if target_chars <= 0 or max_chars < target_chars:
            raise ValueError("invalid chunk size configuration")
        self.target_chars = target_chars
        self.max_chars = max_chars
        canonical = json.dumps(
            {
                "strategy_version": CHUNKING_STRATEGY_VERSION,
                "target_chars": target_chars,
                "max_chars": max_chars,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.identity = hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def strategy_version(self) -> str:
        return CHUNKING_STRATEGY_VERSION

    def chunk(self, text: str, *, base_seed: int) -> list[TextChunk]:
        if not text:
            raise ValueError("text must not be empty")
        paragraphs = {match.end() for match in _PARAGRAPH.finditer(text)}
        sentences = {
            match.end()
            for match in _SENTENCE.finditer(text)
            if not self._is_decimal_boundary(text, match.start())
        }
        whitespace = {match.end() for match in _WHITESPACE.finditer(text)}
        chunks: list[TextChunk] = []
        start = 0
        while start < len(text):
            if len(text) - start <= self.max_chars:
                end = len(text)
                boundary = "end"
            else:
                end, boundary = self._choose_boundary(
                    start,
                    len(text),
                    paragraphs,
                    sentences,
                    whitespace,
                )
            if end <= start:
                end = min(start + self.max_chars, len(text))
                boundary = "hard"
            index = len(chunks)
            chunks.append(
                TextChunk(
                    index=index,
                    start_char=start,
                    end_char=end,
                    text=text[start:end],
                    boundary=boundary,
                    seed=derive_chunk_seed(base_seed, index),
                )
            )
            start = end
        if "".join(chunk.text for chunk in chunks) != text:
            raise RuntimeError("chunker failed to preserve source text")
        return chunks

    def _choose_boundary(
        self,
        start: int,
        text_length: int,
        paragraphs: set[int],
        sentences: set[int],
        whitespace: set[int],
    ) -> tuple[int, str]:
        target = min(start + self.target_chars, text_length)
        maximum = min(start + self.max_chars, text_length)
        minimum_preferred = start + max(1, self.target_chars // 2)
        for positions, reason in (
            (paragraphs, "paragraph"),
            (sentences, "sentence"),
        ):
            before = [p for p in positions if minimum_preferred <= p <= target]
            if before:
                return max(before), reason
        for positions, reason in (
            (paragraphs, "paragraph"),
            (sentences, "sentence"),
        ):
            after = [p for p in positions if target < p <= maximum]
            if after:
                return min(after), reason
        spaces = [p for p in whitespace if start < p <= maximum]
        if spaces:
            return max(spaces), "fallback"
        return maximum, "hard"

    @staticmethod
    def _is_decimal_boundary(text: str, position: int) -> bool:
        return (
            text[position] == "."
            and position > 0
            and position + 1 < len(text)
            and text[position - 1].isdigit()
            and text[position + 1].isdigit()
        )
