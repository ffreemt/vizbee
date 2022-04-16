"""Align ns.df (df in ns)."""
# pylint: disable=invalid-name, line-too-long

from itertools import zip_longest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from logzero import logger

from pyvizbee.align_df_sliced import align_df_sliced
from pyvizbee.locate_index_pairs import locate_index_pairs


def align_ns_df(
    df_: pd.DataFrame,
    model: TfidfVectorizer,
) -> pd.DataFrame:
    """Align ns.df (df in ns).

    Args:
        df_: pd.DataFrame, columns = ['text1', 'text2', 'metric'], df_.index.name = 'seq'
        model: sklearn.feature_extraction.text.TfidfVectorizer on Chinese text, convert en2zh(process_en) for English text
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
    df_sl = [*zip_longest((0,) + r, l)]

    # [(0, 2), (3, 4), (5, 10), (11, 18), (19, 25), (26, 31), (32, 33), (35, None)]

    # prep pd.concat
    df_new = []
    for sl0, sl1 in zip(df_sl, sl):
        df_new.append(df_[slice(*sl0)])

        _ = align_df_sliced(df_, sl1, model)
        df_new.append(_)

    # collect possible last segment
    df_new.append(df_[slice(*df_sl[-1])])

    _ = pd.concat(df_new, ignore_index=True)
    _.index.name = "seq"

    return _
