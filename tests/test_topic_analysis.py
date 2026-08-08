"""Unit tests for ``real_time_topic_analysis_for_X_previously_Tweeter_.ipynb``."""

import pytest

from tests.notebook_loader import load_definitions

NOTEBOOK = "real_time_topic_analysis_for_X_previously_Tweeter_.ipynb"
STOP_WORDS = {"the", "and", "for", "kenya", "https", "co", "rt", "amp"}

CLEANING = load_definitions(
    NOTEBOOK, ["clean_text"], cell_index=5, inject={"stop_words": STOP_WORDS}
)
ENTITIES = load_definitions(NOTEBOOK, ["merge_and_format_entities"], cell_index=19)


class TestCleanText:
    def test_lowercases_and_tokenizes(self):
        assert CLEANING["clean_text"]("Nairobi MARATHON") == ["nairobi", "marathon"]

    def test_drops_urls_and_punctuation(self):
        assert CLEANING["clean_text"]("great news!! http://t.co/abc, really") == [
            "great",
            "news",
            "really",
        ]

    def test_drops_stop_words_and_short_tokens(self):
        assert CLEANING["clean_text"]("the Kenya economy is up") == ["economy"]

    @pytest.mark.parametrize("text", ["", "   ", "rt amp co"])
    def test_returns_an_empty_list_for_uninformative_text(self, text):
        assert CLEANING["clean_text"](text) == []

    def test_keeps_token_order_and_duplicates(self):
        assert CLEANING["clean_text"]("tax tax reform") == ["tax", "tax", "reform"]


class TestMergeAndFormatEntities:
    @pytest.mark.parametrize(
        "tokens, expected",
        [
            (["sabastian", "sawe"], ["Sabastian", "Sawe"]),
            (["sawe", "sabastian"], ["Sabastian", "Sawe"]),
            (["record", "breaker"], ["Record", "Breaker"]),
            (["mountain", "bongo"], ["Mountain", "Bongo"]),
        ],
    )
    def test_known_entities_are_normalised(self, tokens, expected):
        assert ENTITIES["merge_and_format_entities"](tokens) == expected

    def test_unknown_tokens_pass_through_unchanged(self):
        tokens = ["nairobi", "marathon", "results"]
        assert ENTITIES["merge_and_format_entities"](tokens) == tokens

    def test_matching_is_case_insensitive(self):
        assert ENTITIES["merge_and_format_entities"](["SABASTIAN", "SAWE"]) == [
            "Sabastian",
            "Sawe",
        ]

    def test_entities_are_normalised_inside_a_longer_phrase(self):
        assert ENTITIES["merge_and_format_entities"](
            ["new", "record", "breaker", "today"]
        ) == ["new", "Record", "Breaker", "today"]

    def test_empty_input(self):
        assert ENTITIES["merge_and_format_entities"]([]) == []
