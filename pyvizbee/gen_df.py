"""Gen df."""
from pathlib import Path
import os
from itertools import zip_longest
import pandas as pd
from logzero import logger

from fast_scores.en2zh import en2zh
from fast_scores.process_en import process_en
from fast_scores.gen_model import gen_model
from fastlid import fastlid

from pyvizbee.ns import ns
from pyvizbee.filevalue2list import filevalue2list
from pyvizbee.loadtext import loadtext

VIZBEE_DEV = os.environ.get("VIZBEE_DEV")


def gen_df():
    """Gen df (pandas.DataFrame) from file1.value/file2.value.

    _ =  [elm for elm in ns.df.text1.to_list() if elm.strip()]
    cmat = gen_cmat(_, ns.df.text2.to_list())
    """
    # global file1, file2

    list1 = filevalue2list(ns.file1.value)
    list2 = filevalue2list(ns.file2.value)

    logger.debug("list1[:10]: %s", list1[:10])
    logger.debug("list2[:10]: %s", list2[:10])

    if VIZBEE_DEV:
        # set default during dev
        if not list1:
            text1 = loadtext(Path(__file__).parent / "test-en.txt")
            list1 = [elm.strip() for elm in text1.splitlines() if elm.strip()]
        if not list2:
            text2 = loadtext(Path(__file__).parent / "test-zh.txt")
            list2 = [_.strip() for _ in text2.splitlines() if _.strip()]

    ns.df = pd.DataFrame(
        zip_longest(list1, list2, [], fillvalue=""),
        columns=["text1", "text2", "metric"],
    )

    # ns.df.insert(0, "seq", range(1, 1 + len(ns.df)))
    # ns.df = ns.df.set_index("seq")
    # ns.df.index = [*range(1, 1 + len(ns.df))]
    # ns.df.index += 1
    ns.df.index.name = "seq"

    # update ns.lang1, ns.lang2; default en, zh
    fastlid.set_languages = ["en", "zh"]
    try:
        ns.lang1, *_ = fastlid(list1)
        ns.lang2, *_ = fastlid(list2)
    except Exception:
        logger.exception(" ns.langx, *_ = fastlid(listx)")
        logger.info("Continue and hope for the best...")
    # also gen model

    vec1, vec2 = list1, list2
    if ns.lang1 in ["en"]:
        vec1 = en2zh(process_en(list1))
    if ns.lang2 in ["en"]:
        vec2 = en2zh(process_en(list2))
    try:
        ns.model = gen_model(vec1 + vec2)
    except Exception:
        logger.exception("ns.model = gen_model(vec1 + vec2)")
        logger.info("We pretend nothing happens and hope for the best.")

    return ns.df
