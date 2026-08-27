import pytest
from voice_service.services.text_chunker import TextChunker, derive_chunk_seed


def assert_preserved(text: str, chunker: TextChunker) -> None:
    chunks = chunker.chunk(text, base_seed=123)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(len(chunk.text) <= chunker.max_chars for chunk in chunks)
    assert chunks == chunker.chunk(text, base_seed=123)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_short_text_is_one_chunk() -> None:
    chunks = TextChunker(target_chars=20, max_chars=30).chunk("Texto curto.", base_seed=8)
    assert [chunk.text for chunk in chunks] == ["Texto curto."]
    assert chunks[0].boundary == "end"


@pytest.mark.parametrize(
    "text",
    [
        "Primeira frase. Segunda frase! Terceira frase?",
        "Primeiro paragrafo.\n\nSegundo paragrafo com mais texto.",
        "Uma frase enorme " + "palavra " * 40 + "fim.",
        "A versao 3.14 custa 10.50 reais. Depois continua.",
        "O heroi acende o sinal Aard. Coracao, acao e consequencia.",
        "token" * 30,
        "  espaco inicial.  Espaco final.  ",
    ],
)
def test_chunking_is_lossless_bounded_and_deterministic(text: str) -> None:
    assert_preserved(text, TextChunker(target_chars=35, max_chars=55))


def test_paragraph_and_sentence_boundaries_are_preferred() -> None:
    text = "Frase inicial suficientemente longa. Outra frase.\n\nNovo paragrafo final."
    chunks = TextChunker(target_chars=35, max_chars=55).chunk(text, base_seed=1)
    assert any(chunk.boundary in {"sentence", "paragraph"} for chunk in chunks[:-1])


def test_chunk_seed_derivation_is_stable_and_distinct() -> None:
    assert derive_chunk_seed(77, 0) == derive_chunk_seed(77, 0)
    assert derive_chunk_seed(77, 0) != derive_chunk_seed(77, 1)
    assert 1 <= derive_chunk_seed(77, 0) <= 2**31 - 1


def test_chunking_identity_tracks_generation_affecting_limits() -> None:
    baseline = TextChunker(target_chars=250, max_chars=400)
    same = TextChunker(target_chars=250, max_chars=400)
    changed = TextChunker(target_chars=251, max_chars=400)
    assert baseline.strategy_version == "qwen3-tts-chunking-v2"
    assert baseline.identity == same.identity
    assert baseline.identity != changed.identity


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        TextChunker(target_chars=20, max_chars=30).chunk("", base_seed=1)
