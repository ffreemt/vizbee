"""Gen aligned pairs for given list1 list2 and aset."""
# pylint: disable=invalid-name
from typing import List
from logzero import logger


def gen_pairs(list1: List[str], list2: List[str], aset: List[str]) -> List[str]:
    """Gen aligned pairs for given list1 list2 and aset."""
    res = []
    for elm in aset:
        try:
            ix, iy, val, *_ = elm
        except Exception as e:
            logger.exception(e)
            raise

        term = []
        try:
            iy = int(iy)  # note iy, not ix
            _ = list1[iy]
        except Exception:
            _ = ""
        term.append(_)

        try:
            ix = int(ix)
            _ = list2[ix]
        except Exception:
            _ = ""
        term.append(_)

        try:
            val = float(val)
        except Exception:
            val = ""
        term.append(val)

        res.append(term)

    return res
