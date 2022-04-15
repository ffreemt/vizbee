"""Test test_align_df_sliced."""
# pylint: disable=invalid-name

import tests.ns_df as ns_df
from pyvizbee.align_df_sliced import align_df_sliced
from pyvizbee.locate_index_pairs import locate_index_pairs

df_ = ns_df.df_.copy()
model = ns_df.model

assert len(df_) > 30


def test_align_df_sliced():
    """Test align_df_sliced."""
    lst = df_.metric[df_.metric.replace("", float("-inf")) < 0].index.to_list()
    sl = locate_index_pairs(lst)

    assert sl[3] == (18, 19)

    _ = align_df_sliced(df_, sl[3], model)
    assert _.shape == (1, 3)
    assert _.metric[0] > 0.2


def test_align_df_sliced02():
    """Test align_df_sliced 0.2."""
    # float("-inf") or use -np.inf
    # lst02a = df_.metric[df_.metric.replace("", -np.inf) < 0.2].index.to_list()
    lst02 = df_.metric[df_.metric.replace("", float("-inf")) < 0.2].index.to_list()

    # assert lst02 == lst02a

    sl02 = locate_index_pairs(lst02)

    # assert sl[3] == (18, 19)
    assert sl02[-1] == (31, 35)

    _ = align_df_sliced(df_, sl02[-1], model)
    assert _.shape == (4, 3)
    assert _.metric[0] > 0.1
