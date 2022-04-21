"""Test sent_align_df_sliced."""
from itertools import zip_longest
from tests.ns_df import df_

from pyvizbee.locate_index_pairs import locate_index_pairs
from pyvizbee.not_docked_indices import not_docked_indices
from pyvizbee.sent_align_df_sliced import sent_align_df_sliced


def test_sent_align_df_sliced():
    """Test seg_to_sents."""
    lst = not_docked_indices(df_)
    sl = locate_index_pairs(lst)
    # [(2, 3), (4, 5), (10, 11), (18, 19), (25, 26), (31, 32), (33, 35)]

    # refer also to align_ns_df.py
    l, r = zip(*sl)
    docked = [*zip_longest((0,) + r, l)]
    # [(0, 2), (3, 4), (5, 10), (11, 18), (19, 25), (26, 31), (32, 33), (35, None)]

    res = sent_align_df_sliced(df_, (5, 6), "en", "zh")
    # bad one: [(0, '', ''),
    # (1, '', ''),
    # (2, '', ''),
    # (3, 0),
    # (3, 1),
    # ('', 2, ''),
    # ('', 3, '')]
    assert len(res) > 5  # 7

    res = sent_align_df_sliced(df_, (6, 7))
    assert len(res) == 3

    res = sent_align_df_sliced(df_, (7, 8))
    assert len(res) == 7

    # sl[3]: slice_tuple = (18, 19)
    # I commenced again. `Do you intend parting with
    # need to convert `
    res = sent_align_df_sliced(df_, (18, 19))
    assert len(res) == 3

    # (33, 35)
    res = sent_align_df_sliced(df_, (33, 35))
    assert len(res) > 1
