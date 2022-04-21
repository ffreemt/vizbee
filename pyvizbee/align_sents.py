"""Align sents via gale-church."""
# pylint: disable=invalid-name

from typing import List, Tuple  # noqa

import re

# from itertools import tee
# from more_itertools import ilen
from nltk.translate.gale_church import align_blocks

from pyvizbee.amend_avec import amend_avec


def align_sents(lst1: List[str], lst2: List[str]) -> List[Tuple[str, str]]:
    """Align sents.

    >>> lst1, lst2 = ['a', 'bs',], ['aaa', '34', 'a', 'b']
    """
    if isinstance(lst1, str):
        lst1 = [lst1]

    if isinstance(lst2, str):
        lst2 = [lst2]

    src_blocks = [len(re.sub(r"\s+", "", elm)) for elm in lst1]
    tgt_blocks = [len(re.sub(r"\s+", "", elm)) for elm in lst2]

    avec = align_blocks(src_blocks, tgt_blocks)

    len1, len2 = len(lst1), len(lst2)
    # lst1, _ = tee(lst1)
    # len1 = ilen(_)
    # lst2, _ = tee(lst2)
    # len2 = ilen(_)

    amended_avec = amend_avec(avec, len1, len2)

    texts = []
    # for elm in aset:
    # for elm0, elm1 in amended_avec:
    for elm in amended_avec:
        # elm0, elm1, elm2 = elm
        elm0, elm1 = elm[:2]
        _ = []

        # src_text first
        if isinstance(elm0, str):
            _.append("")
        else:
            # _.append(src_text[int(elm0)])
            _.append(lst1[int(elm0)])

        if isinstance(elm1, str):
            _.append("")
        else:
            # _.append(tgt_text[int(elm0)])
            _.append(lst2[int(elm1)])

        _a = """
        if isinstance(elm2, str):
            _.append("")
        else:
            _.append(round(elm2, 2))
        # """
        del _a

        texts.append(tuple(_))

    _ = """
    _ = []
    for elm in texts:
        _.extend(elm)
    return _
    """

    return texts
