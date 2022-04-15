"""Prep ns.df for align/re-align.

tab1 Load Files
"""
# pylint: disable=invalid-name

from pathlib import Path

# import numpy as np
import pandas as pd
import seaborn as sns
from itertools import zip_longest
from pyvizbee.loadtext import loadtext
from types import SimpleNamespace

# cd pipy cd ezbee
# from add_path import add_path
from pyvizbee.add_path import add_path  # noqa

# add_path([r"..\ezbee", r"..\cmat2aset", r"..\fast-scores"])  # noqa

from cmat2aset import cmat2aset
from ezbee.gen_pairs import gen_pairs

from fast_scores.gen_model import gen_model
from fast_scores.gen_cmat import gen_cmat
from fast_scores.en2zh import en2zh
from fast_scores.process_en import process_en
from fastlid import fastlid


sns.set()
sns.set_style("darkgrid")

file_en = "text-en.txt"
file_zh = "text-zh.txt"

DEFAULT_EPS = 10
DEFAULT_MIN_SAMPLES = 6

# workspace namespace: similar to global()
ns = SimpleNamespace(
    **{
        "counter": 0,
        "df": None,
        "df_pane": None,
        "eps": DEFAULT_EPS,  # Two points are considered neighbors if the distance between the two points is below eps
        "min_samples": DEFAULT_MIN_SAMPLES,  # The minimum number of neighbors a given point should have in order to be classified as a core point.
    }
)

# if "__file__" not in globals():  __file__ = "tests\ns_df.py"
text1 = loadtext(Path("pyvizbee", "test-en.txt"))
list1 = [elm.strip() for elm in text1.splitlines() if elm.strip()]

text2 = loadtext(Path("pyvizbee", "test-zh.txt"))
list2 = [_.strip() for _ in text2.splitlines() if _.strip()]

ns.df = pd.DataFrame(
    zip_longest(list1, list2, [], fillvalue=""),
    columns=["text1", "text2", "metric"],
)

ns.df.insert(0, "seq", range(1, 1 + len(ns.df)))
ns.df = ns.df.set_index("seq")

fastlid.set_langues = ['en', 'zh']
lang1, *_ = fastlid(ns.df.text1)
lang2, *_ = fastlid(ns.df.text2)


if lang1 in ['en']:
    model = gen_model(en2zh(process_en(ns.df.text1)) + ns.df.text2.to_list())
else:
    model = gen_model(en2zh(process_en(ns.df.text2)) + ns.df.text1.t_list())

cmat_10 = gen_cmat(list1[:10], list2[:10], model=model)
aset = cmat2aset(cmat_10)
res = gen_pairs(list1[:10], list2[:20], aset)
df10 = pd.DataFrame(res, columns=ns.df.columns)
df10a = df10.append(ns.df.iloc[11:, :])

# pd.concat([df_update, df], axis=0, ignore_index=True)
# or concat assign set_index https://sparkbyexamples.com/pandas/pandas-add-column-to-dataframe/#:~:text=In%20pandas%20you%20can%20add,after%20adding%20a%20new%20column.

df10a.insert(0, "seq", range(1, len(df10a) + 1))
df10a.set_index('seq', inplace=True)

# df10.assign(

cmat = gen_cmat(list1, list2, model=model)
aset = cmat2aset(cmat)

res = gen_pairs(list1, list2, aset)

df_ = pd.DataFrame(res, columns=ns.df.columns)
df_.index.name = "seq"

lst = df_.metric[df_.metric.replace("", float("-inf")) < 0.4].index.to_list()
# [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34]

# In [428]: df_.metric[df_.metric.replace("", float("-inf")) < 0.1].index.to_list()
# Out[428]: [0, 1, 2, 4, 10, 18, 25, 31, 33, 34]
# -> [0:3], 4:5, 10:11, 18:19, 25:26, 31:32, 33:35

# collect items for metrix: "" and <= thr

# align/re-align
