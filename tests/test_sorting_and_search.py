"""Unit tests for the sorting/searching notebooks (``sorts``, ``bubble_search_algos``)."""

import random

import pytest

from tests.notebook_loader import load_definitions

SORTS = load_definitions(
    "sorts.ipynb",
    ["count_sort", "countingSort_for_radix", "radixSort", "bubble_sort"],
)
SEARCH = load_definitions(
    "bubble_search_algos.ipynb", ["linear_search", "bubble_sort"]
)


class TestCountSort:
    def test_empty_input_returns_empty_list(self):
        assert SORTS["count_sort"]([]) == []

    @pytest.mark.parametrize(
        "values",
        [
            [0],
            [1, 0],
            [4, 2, 2, 8, 3, 3, 1, 0, 5, 4],
            [7, 7, 7],
            list(range(20))[::-1],
        ],
    )
    def test_sorts_non_negative_integers(self, values):
        assert SORTS["count_sort"](values) == sorted(values)

    def test_does_not_mutate_input(self):
        values = [3, 1, 2]
        SORTS["count_sort"](values)
        assert values == [3, 1, 2]

    def test_matches_sorted_on_random_inputs(self):
        rng = random.Random(1234)
        for _ in range(25):
            values = [rng.randint(0, 50) for _ in range(rng.randint(1, 40))]
            assert SORTS["count_sort"](values) == sorted(values)


class TestRadixSort:
    def test_counting_sort_for_radix_sorts_by_single_digit(self):
        values = [170, 45, 75, 90, 802, 24, 2, 66]
        SORTS["countingSort_for_radix"](values, 1)
        assert [value % 10 for value in values] == sorted(v % 10 for v in values)

    def test_counting_sort_for_radix_is_stable(self):
        # Equal least significant digits must keep their relative order.
        values = [21, 11, 31]
        SORTS["countingSort_for_radix"](values, 1)
        assert values == [21, 11, 31]

    def test_sorts_in_place(self):
        values = [170, 45, 75, 90, 802, 24, 2, 66]
        assert SORTS["radixSort"](values) is None
        assert values == [2, 24, 45, 66, 75, 90, 170, 802]

    def test_single_element(self):
        values = [5]
        SORTS["radixSort"](values)
        assert values == [5]

    def test_matches_sorted_on_random_inputs(self):
        rng = random.Random(99)
        for _ in range(25):
            values = [rng.randint(0, 10_000) for _ in range(rng.randint(1, 40))]
            expected = sorted(values)
            SORTS["radixSort"](values)
            assert values == expected


class TestInPlaceBubbleSort:
    """``sorts.ipynb`` defines a bubble sort that sorts in place without returning."""

    def test_returns_nothing_and_sorts_in_place(self):
        values = [64, 34, 25, 12, 22, 11, 90]
        assert SORTS["bubble_sort"](values) is None
        assert values == [11, 12, 22, 25, 34, 64, 90]

    @pytest.mark.parametrize("values", [[], [1], [2, 1], [1, 1, 1]])
    def test_edge_cases(self, values):
        expected = sorted(values)
        SORTS["bubble_sort"](values)
        assert values == expected

    def test_matches_sorted_on_random_inputs(self):
        rng = random.Random(7)
        for _ in range(25):
            values = [rng.randint(-50, 50) for _ in range(rng.randint(0, 30))]
            expected = sorted(values)
            SORTS["bubble_sort"](values)
            assert values == expected


class TestBubbleSort:
    def test_returns_sorted_list(self):
        assert SEARCH["bubble_sort"]([64, 34, 25, 12, 22, 11, 90]) == [
            11,
            12,
            22,
            25,
            34,
            64,
            90,
        ]

    def test_sorts_in_place_and_returns_same_object(self):
        values = [3, 2, 1]
        result = SEARCH["bubble_sort"](values)
        assert result is values
        assert values == [1, 2, 3]

    @pytest.mark.parametrize("values", [[], [1], [1, 2, 3], [2, 1]])
    def test_edge_cases(self, values):
        assert SEARCH["bubble_sort"](list(values)) == sorted(values)

    def test_handles_duplicates_and_negatives(self):
        values = [0, -5, 3, -5, 3]
        assert SEARCH["bubble_sort"](values) == [-5, -5, 0, 3, 3]


class TestLinearSearch:
    def test_finds_first_index_of_target(self):
        assert SEARCH["linear_search"]([4, 7, 7, 1], 7) == 1

    def test_returns_minus_one_when_absent(self):
        assert SEARCH["linear_search"](list(range(10)), 10) == -1

    def test_empty_list(self):
        assert SEARCH["linear_search"]([], 1) == -1

    @pytest.mark.parametrize("target", [0, 5, 9])
    def test_agrees_with_list_index(self, target):
        values = list(range(10))
        assert SEARCH["linear_search"](values, target) == values.index(target)
