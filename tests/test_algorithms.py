import random

import pytest

from will_utils.algorithms import (
    binary_search,
    bubble_sort,
    count_sort,
    linear_search,
    radix_sort,
    report_search,
)

CASES = [
    [],
    [1],
    [4, 2, 2, 8, 3, 3, 1, 0, 5, 4],
    [170, 45, 75, 90, 802, 24, 2, 66],
    [random.randint(0, 100) for _ in range(100)],
]


@pytest.mark.parametrize("data", CASES)
@pytest.mark.parametrize("sort", [bubble_sort, count_sort, radix_sort])
def test_sorts_match_builtin(sort, data):
    assert list(sort(list(data))) == sorted(data)


def test_bubble_sort_sorts_in_place():
    data = [3, 1, 2]
    assert bubble_sort(data) is data
    assert data == [1, 2, 3]


@pytest.mark.parametrize("sort", [count_sort, radix_sort])
def test_negative_values_rejected(sort):
    with pytest.raises(ValueError):
        sort([1, -2, 3])


@pytest.mark.parametrize("search", [linear_search, binary_search])
def test_search_finds_and_misses(search):
    data = list(range(1000))
    assert search(data, 778) == 778
    assert search(data, 1000) == -1


def test_report_search_prints_result(capsys):
    assert report_search([1, 2, 3], 2) == 1
    assert "found at index 1" in capsys.readouterr().out
    assert report_search([1, 2, 3], 9) == -1
    assert "not found" in capsys.readouterr().out
