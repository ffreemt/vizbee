"""Gen T or F list according to df.metric is "" and < thr."""
from typing import List
import pandas as pd

from logzero import logger


def not_docked_indices(
    df_: pd.DataFrame,
    threshold: float = 0,
) -> List[int]:
    """Gen T or F list according to df.metric is "" and < thr.

    Args:
        df_: pd.DataFrame with metric column of float and "".

    Returns:
        list of integers, df_.metric's indices at with value smaller than threshold or is "".
    """
    try:
        lst = df_.metric[df_.metric.replace("", float("-inf")) < threshold].index.to_list()
    except Exception:
        logger.exception("")
        raise

    return lst
