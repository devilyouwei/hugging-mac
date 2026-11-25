import numpy as np
import pytest
from hugging_mac_conversion.models.zipformer.publication.spotter import (
    Array,
    Beam,
    Hit,
    Keyword,
    read_keywords,
)


def test_keyword_file_accepts_bilingual_tokens_and_rejects_duplicates(tmp_path):
    vocabulary = tmp_path / "tokens.txt"
    vocabulary.write_text("<blk> 0\n<unk> 2\nS 3\nT 4\nzh 5\nōu 6\n", encoding="utf-8")
    phrases = tmp_path / "keywords.txt"
    phrases.write_text("S T @STOP\nzh ōu :1.4 #0.5 @周\n", encoding="utf-8")
    assert read_keywords(phrases, vocabulary) == {
        (3, 4): Keyword("STOP", 1.0, 0.25),
        (5, 6): Keyword("周", 1.4, 0.5),
    }
    phrases.write_text("S T @STOP\nS T @DUPLICATE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique nonblank"):
        read_keywords(phrases, vocabulary)


def test_beam_waits_for_blank_and_keeps_timestamp_across_chunks():
    def predict(part: str, inputs: tuple[Array, ...]) -> tuple[Array, ...]:
        if part == "decoder":
            return (np.zeros((1, 1), dtype=np.float32),)
        logits = np.full((1, 7), -20, dtype=np.float32)
        logits[0, int(inputs[0][0, 0])] = 20
        return (logits,)

    beam = Beam(predict, {(3, 4): Keyword("STOP", 1.0, 0.25)})
    assert beam.step(np.array([[3]]), trailing_blanks=1) is None
    assert beam.step(np.array([[4]]), trailing_blanks=1) is None
    assert beam.step(np.array([[0]]), trailing_blanks=1) is None
    assert beam.step(np.array([[0]]), trailing_blanks=1) == Hit("STOP", (0, 1))
    assert beam.step(np.array([[0]]), trailing_blanks=1) is None
