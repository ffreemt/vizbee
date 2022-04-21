"""Align ns.df (df in ns)."""
# pylint: disable=invalid-name, line-too-long

from itertools import zip_longest

import pandas as pd
from logzero import logger

from pyvizbee.locate_index_pairs import locate_index_pairs
from pyvizbee.sent_align_df_sliced import sent_align_df_sliced


def sent_align_ns_df(
    df_: pd.DataFrame,
    lang1: str = "en",
    lang2: str = "zh",
) -> pd.DataFrame:
    """Align ns.df (df in ns).

    Args:
        df_: pd.DataFrame, columns = ['text1', 'text2', 'metric'], df_.index.name = 'seq'

    Returns:
        aligned DataFrame with the same columns as df_.
    """
    try:
        lst = df_.metric[df_.metric.replace("", float("-inf")) < 0].index.to_list()
        sl = locate_index_pairs(lst)
    except Exception:
        logger.exception("")
        raise

    # [(2, 3), (4, 5), (10, 11), (18, 19), (25, 26), (31, 32), (33, 35)]

    l, r = zip(*sl)
    # l: (2, 4, 10, 18, 25, 31, 33)
    # r: (3, 5, 11, 19, 26, 32, 35)

    # [*zip_longest((0,) + r, l, fillvalue=None)]
    docked = [*zip_longest((0,) + r, l)]

    # [(0, 2), (3, 4), (5, 10), (11, 18), (19, 25), (26, 31), (32, 33), (35, None)]

    # prep pd.concat
    df_new = []
    for sl0, sl1 in zip(docked, sl):
        # df_new.append(df_[slice(*sl0)])
        # process docked rows sl0
        start, end = sl0
        for elm in range(start, end):
            _ = sent_align_df_sliced(df_, (elm, elm + 1), lang1, lang2)
            df_new.append(_)

        # undocked part
        _ = sent_align_df_sliced(df_, sl1, lang1, lang2)
        df_new.append(_)

    # collect possible last segment
    start, end = docked[-1]
    end = len(df_)
    # df_new.append(df_[slice(*docked[-1])])
    for elm in range(start, end):
        _ = sent_align_df_sliced(df_, (elm, elm + 1), lang1, lang2)
        df_new.append(_)

    _ = pd.concat(df_new, ignore_index=True)
    _.index.name = "seq"

    return _
