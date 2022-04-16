"""Prep namespace ns."""
# pylint: disable=invalid-name
from pathlib import Path
from types import SimpleNamespace
import panel as pn
import joblib

file1 = pn.widgets.FileInput(accept=".txt,.csv,.tsv")
file2 = pn.widgets.FileInput(accept=".txt,.csv,.tsv")

default_eps = 10
default_min_samples = 6
model_test_zh = joblib.load(Path(__file__).parent / "model_test_zh.lzma")

# workspace namespace: similar to globals()
ns = SimpleNamespace(
    **{
        "counter": 0,
        "df": None,
        "df_pane": None,
        "eps": default_eps,  # Two points are considered neighbors if the distance between the two points is below eps
        "min_samples": default_min_samples,  # The minimum number of neighbors a given point should have in order to be classified as a core point.
        "lang1": "en",
        "lang2": "zh",
        "file1": file1,
        "file2": file2,
        "list1": [""],
        "list2": [""],
        "model": model_test_zh,  # default testing model
    }
)
