"""Align df_ given slice(pairs) model."""
from typing import Tuple, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from logzero import logger

from fast_scores.gen_cmat import gen_cmat
from cmat2aset import cmat2aset
from pyvizbee.gen_pairs import gen_pairs


def align_df_sliced(
    df_: pd.DataFrame,
    slice_tuple: Tuple[Optional[int], Optional[int]],
    model: TfidfVectorizer,
) -> pd.DataFrame:
    """Align df_ given sliced (pairs) and a model.

    Args:
        df_: dataframe
        slice_tuple: integer two-tuple
        model: for calculating cmat
    Returns:
        dataframe with the same columns as df_
    e.g., align_df_slice(df_, sl[3], model)
    """
    lst1 = df_[slice(*slice_tuple)].text1.to_list()
    lst2 = df_[slice(*slice_tuple)].text2.to_list()
    try:
        cmat = gen_cmat(
            lst1,
            lst2,
            model,
        )
    except Exception:
        logger.exception("gen_cmat")
        raise

    try:
        aset = cmat2aset(cmat)
    except Exception:
        logger.exception("cmat2aset")
        raise

    res = gen_pairs(lst1, lst2, aset)

    _ = pd.DataFrame(res, columns=df_.columns)
    _.index.name = df_.index.name

    return _
