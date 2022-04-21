"""Align df_ given slice(pairs) model."""
from itertools import zip_longest
from typing import Optional, Tuple

import pandas as pd
from logzero import logger
# from seg_text import seg_text

from pyvizbee.align_sents import align_sents
from pyvizbee.seg_text_fix import seg_text_fix as seg_text


def sent_align_df_sliced(
    df_: pd.DataFrame,
    slice_tuple: Tuple[Optional[int], Optional[int]],
    lang1: str = "en",
    lang2: str = "zh",
) -> pd.DataFrame:
    """Sent-Align df_ given sliced (pairs).

    Args:
        df_: dataframe
        slice_tuple: integer two-tuple
        lang1: default "en", launguage of text1 ("en", "zh", etc)
        lang2: default "zh", launguage of text2 ("en", "zh", etc)

    Returns:
        dataframe with the same columns as df_
        e.g., sent_align_df_slice(df_, sl[3])
    """
    lst1 = df_[slice(*slice_tuple)].text1.to_list()
    lst2 = df_[slice(*slice_tuple)].text2.to_list()
    metric = df_[slice(*slice_tuple)].metric.to_list()

    # seg_text to sents and align resultant sents

    try:
        list1 = seg_text(lst1, lang1)
    except Exception:
        logger.exception("seg_text lst1")
        raise
    try:
        list2 = seg_text(lst2, lang2)
    except Exception:
        logger.exception("seg_text lst2")
        raise

    try:
        sents_ali = align_sents(list1, list2)
    except Exception:
        logger.exception("align_sents")
        raise

    try:
        left, right = zip(*sents_ali)
    except Exception:
        logger.exception("left, right = zip(*sents_ali)")
        logger.debug("list1: %s, list2: %s", list1, list2)
        logger.debug("sents_ali: %s", sents_ali)
        logger.warning(" return the original")

        if metric:
            metric0 = metric[0]
        else:
            metric0 = ""
        res = zip_longest(list1, list2, [metric0], fillvalue="")
        _ = pd.DataFrame(res, columns=df_.columns)
        _.index.name = df_.index.name

        return _
        # raise

    if metric:
        metric0 = metric[0]
    else:
        metric0 = ""

    res = zip_longest(left, right, [metric0], fillvalue="")

    _ = pd.DataFrame(res, columns=df_.columns)
    _.index.name = df_.index.name

    return _
