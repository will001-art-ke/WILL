"""Classic sorting and searching algorithms used by the algorithm notebooks."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence


def bubble_sort(arr: MutableSequence[int]) -> MutableSequence[int]:
    """Sort ``arr`` in place with bubble sort and return it."""
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def count_sort(arr: Sequence[int]) -> list[int]:
    """Stable counting sort for non-negative integers."""
    if not arr:
        return []
    if min(arr) < 0:
        raise ValueError("count_sort only supports non-negative integers")

    count = [0] * (max(arr) + 1)
    for num in arr:
        count[num] += 1
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    output = [0] * len(arr)
    for num in reversed(arr):
        count[num] -= 1
        output[count[num]] = num
    return output


def _counting_sort_by_digit(arr: MutableSequence[int], exp: int) -> None:
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for num in arr:
        count[(num // exp) % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        count[digit] -= 1
        output[count[digit]] = arr[i]
    arr[:] = output


def radix_sort(arr: MutableSequence[int]) -> MutableSequence[int]:
    """Sort non-negative integers in place with LSD radix sort and return them."""
    if not arr:
        return arr
    if min(arr) < 0:
        raise ValueError("radix_sort only supports non-negative integers")
    exp = 1
    largest = max(arr)
    while largest // exp > 0:
        _counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr


def linear_search(arr: Sequence, target) -> int:
    """Index of ``target`` in ``arr`` scanning left to right, or -1 if absent."""
    for i, value in enumerate(arr):
        if value == target:
            return i
    return -1


def binary_search(arr: Sequence, target) -> int:
    """Index of ``target`` in a sorted ``arr``, or -1 if absent."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def report_search(arr: Sequence, target, search=linear_search) -> int:
    """Run a search and print the found/not-found message the notebooks use."""
    index = search(arr, target)
    if index != -1:
        print(f"Element {target} found at index {index}")
    else:
        print(f"Element {target} not found in the list")
    return index
