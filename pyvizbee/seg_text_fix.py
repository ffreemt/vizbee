"""Fix seg_text' `A problem."""
# pylint: disable=duplicate-code
from typing import List, Optional, Union
import re

from seg_text import seg_text
from logzero import logger


def seg_text_fix(
    lst: Union[str, List[str]],
    lang: Optional[str] = None,
    maxlines: int = 1000,
    extra: Optional[str] = None,
) -> List[str]:
    r"""Fix seg_text.seg_text's \s`[A-Z] problem."""
    if isinstance(lst, str):
        lst_ = [lst]
    else:
        lst_ = lst[:]

    lst_ = [re.sub(r"(\s+)(`)([A-D])", r"\1'\3", elm) for elm in lst_]

    try:
        res = seg_text(lst_, lang=lang, maxlines=maxlines, extra=extra)
    except Exception:
        logger.exception(" ")
        raise

    return res
