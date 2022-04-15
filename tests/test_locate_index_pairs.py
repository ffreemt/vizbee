"""Test locate_index_pairs."""
from typing import List
from pyvizbee.locate_index_pairs import locate_index_pairs


def test_locate_range10():
    """Test range(10)."""
    assert locate_index_pairs(range(10)) == [(0, 10)]


def test_locate_range_lst():
    """Test real example."""
    lst = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34]

    assert locate_index_pairs(lst) == [(0, 5), (7, 13), (14, 24), (25, 28), (29, 35)]

    # lst = df_.metric[df_.metric.replace("", float("-inf")) < 0.].index.to_list()
    lst = [2, 4, 10, 18, 25, 31, 33, 34]
    assert locate_index_pairs(lst) == [(2, 3), (4, 5), (10, 11), (18, 19), (25, 26), (31, 32), (33, 35)]
