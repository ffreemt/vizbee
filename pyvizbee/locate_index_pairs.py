"""Collect consecutie indices for slice to be used in df[slice()]."""
# pylint: disable=invalid-name
from typing import Iterable, List, Tuple, Optional, Union


def locate_index_pairs(
    lst: Union[List[int], Iterable]
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Collect consecutie indices for slice to be used in df[slice()].

    lst: List of integers.
    """
    if not lst:
        return [(None, None)]
        # [][slice(None, None)] == []

    # just in case lst is not sorted: dedup and sort
    lst = sorted(set(lst))

    res = []

    buf = lst[0]  # non-decreasing
    p0 = buf
    for _ in lst[1:]:
        if _ == buf + 1:
            buf = _
        else:
            res.append((p0, buf + 1))
            buf = _
            p0 = buf

    # handle the tail: or
    # res.append((p0, None))
    res.append((p0, buf + 1))

    return res
