"""Test test_align_ns_df."""
# pylint: disable=invalid-name

import tests.ns_df as ns_df
from pyvizbee.align_ns_df import align_ns_df
# from pyvizbee.locate_index_pairs import locate_index_pairs

df_ = ns_df.df_.copy()
model = ns_df.model

assert len(df_) > 30


def test_ns_df():
    """Test ns.df in ns_df."""
    ns = ns_df.ns
    # lst_ns = ns.df.metric[ns.df.metric.replace("", float("-inf")) < 0].index.to_list()
    # sl_ns = locate_index_pairs(lst_ns)

    _ = align_ns_df(ns.df, model)
    assert _.metric.to_list()[-4] > 0.4  # 0.5


def test_align_ns_df():
    """Test align_ns_df."""
    # lst = df_.metric[df_.metric.replace("", float("-inf")) < 0].index.to_list()
    # sl = locate_index_pairs(lst)

    _ = align_ns_df(df_, model)
    assert _.metric.to_list()[28] > 0.4  # .48
