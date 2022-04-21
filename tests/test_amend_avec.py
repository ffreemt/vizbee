"""Test amend_avec."""
from itertools import count, zip_longest
from pyvizbee.amend_avec import amend_avec


def test_amend_avec():
    """Test amend_avec."""
    assert not amend_avec([])
    assert amend_avec([], 3, 2) == [(0, 0), (1, 1), (2, "")]


def test_amend_avec1():
    """Test avec."""
    avec = [
        (0, 0),
        (1, 0),
        (3, 1),
        (4, 1),
        (9, 2),
        (10, 2),
        (13, 3),
        (14, 3),
        (15, 4),
        (16, 4),
        (22, 5),
        (23, 5),
        (27, 6),
        (28, 6),
    ]
    set0, set1 = zip(*amend_avec(avec))
    set0_ = list(set(set0))
    set1_ = list(set(set1))
    assert [*range(29)] == [elm for elm in sorted(set0_, key=set0.index) if elm != ""]
    assert [*range(7)] == [elm for elm in sorted(set1_, key=set1.index) if elm != ""]

    assert [*range(29)] == [elm for elm in dict(zip(set0, count())) if elm != ""]
    assert [*range(7)] == [elm for elm in dict(zip(set1, count())) if elm != ""]

    assert [*range(29)] == [elm for elm in dict(zip_longest(set0, [])) if elm != ""]
    assert [*range(7)] == [elm for elm in dict(zip_longest(set1, [])) if elm != ""]
