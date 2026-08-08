"""Unit tests for the text preprocessing helpers of ``sentiment_analysis.ipynb``."""

import pytest

from tests.notebook_loader import load_definitions

STOP_WORDS = {"the", "a", "is", "and", "to", "of", "i", "it"}

SENTIMENT = load_definitions(
    "sentiment_analysis.ipynb",
    ["preprocess_text"],
    cell_index=16,
    inject={"stop_words": STOP_WORDS},
)


class FakeLemmatizer:
    """Minimal stand-in for ``nltk``'s ``WordNetLemmatizer``."""

    def lemmatize(self, word):
        return word[:-1] if word.endswith("s") else word


LEMMAS = load_definitions(
    "sentiment_analysis.ipynb",
    ["lemmatize_tokens"],
    cell_index=19,
    inject={"lemmatizer": FakeLemmatizer()},
)


class TestPreprocessText:
    def test_lowercases_the_input(self):
        assert SENTIMENT["preprocess_text"]("HELLO World") == "hello world"

    @pytest.mark.parametrize(
        "text",
        ["visit http://example.com now", "visit https://example.com now",
         "visit www.example.com now"],
    )
    def test_removes_urls(self, text):
        assert SENTIMENT["preprocess_text"](text) == "visit now"

    def test_removes_mentions_and_hashtags(self):
        assert SENTIMENT["preprocess_text"]("@alice loves #python code") == (
            "loves code"
        )

    def test_removes_punctuation_numbers_and_extra_whitespace(self):
        assert SENTIMENT["preprocess_text"]("Wow!!!  100%   great,   day?") == (
            "wow great day"
        )

    def test_removes_stop_words(self):
        assert SENTIMENT["preprocess_text"]("the movie is a masterpiece") == (
            "movie masterpiece"
        )

    def test_empty_and_stop_word_only_input(self):
        assert SENTIMENT["preprocess_text"]("") == ""
        assert SENTIMENT["preprocess_text"]("the a is") == ""

    def test_is_idempotent(self):
        once = SENTIMENT["preprocess_text"]("The @user said: visit #now 2 times!")
        assert SENTIMENT["preprocess_text"](once) == once


class TestLemmatizeTokens:
    def test_lemmatizes_every_token(self):
        assert LEMMAS["lemmatize_tokens"](["cats", "dogs", "run"]) == [
            "cat",
            "dog",
            "run",
        ]

    def test_empty_input(self):
        assert LEMMAS["lemmatize_tokens"]([]) == []

    def test_preserves_order_and_length(self):
        tokens = ["apples", "banana", "cherries"]
        result = LEMMAS["lemmatize_tokens"](tokens)
        assert len(result) == len(tokens)
        assert result[1] == "banana"
