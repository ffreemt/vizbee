"""vizbee based on holoviz panel.

set VIZBEE_DEV=1
from add_path import add_path

"""
# pylint: disable=invalid-name, global-statement, unused-argument, unused-import

import os
from pathlib import Path
from itertools import zip_longest, count

# import io
# import logging
# from pathlib import Path
import time
from textwrap import dedent
from types import SimpleNamespace
from typing import List

import cchardet as chardet

# import numpy as np
import pandas as pd
import panel as pn
import param

# from bokeh.models.widgets.tables import CheckboxEditor, NumberEditor, SelectEditor, TextEditor, StringEditor
from bokeh.models.widgets.tables import TextEditor

from unsync import unsync

# import environs
from icecream import ic  # install as ic_install
import logzero
from logzero import logger

from pyvizbee.loglevel import loglevel
from pyvizbee.loadtext import loadtext

logzero.loglevel(loglevel())

logger.debug(" debug: %s", __file__)
logger.info(" info: %s", __file__)

# ic_install()
ic.configureOutput(
    includeContext=True,
    # outputFunction=logger.info,
    outputFunction=logger.debug,
)
ic.enable()
# ic.disable()  # to turn off

VIZBEE_DEV = os.environ.get("VIZBEE_DEV")
if VIZBEE_DEV:
    from icecream import ic
    from add_path import add_path
    add_path([
        r"..\ezbee",
        r"..\cmat2aset",
        r"..\fastscores",
    ])
    try:
        ic(" importing ezbee")
        import ezbee
    except Exception:
        ic(" import ezbee erro")

from fast_scores.get_cmat import gen_cmat

# pn.config.sizing_mode = "stretch_both"
pn.config.sizing_mode = "stretch_width"
pn.extension("tabulator")

file1 = pn.widgets.FileInput(accept=".txt,.csv,.tsv")
file2 = pn.widgets.FileInput(accept=".txt,.csv,.tsv")

# df: pd.DataFrame = None
# df_pane: pn.pane.DataFrame = None

s_btn_load = pn.widgets.Button(name="Load Files")
s_btn_align = pn.widgets.Button(name="Align")
s_btn_about = pn.widgets.Button(name="About")


# workspace namespace: similar to global()
ns = SimpleNamespace(**{
    "counter": 0,
    "df": None,
    "df_pane": None,
})

about_info = dedent(
    """
    ## Visbee Aligner

    from mu's keyborad in cyberspace (join qq group **316287378** or visit the bumblebee forum [bumblebee.freeforums.net](https://bumblebee.freeforums.net/board/1/general-discussion) to be kept informed and for questions and feedback)
    """
).strip()
_ = """
    <hr>

    Two large model files (1.2G and 500M) will be downloaded (from
    <a href="https://huggingface.co/datasets/mikeee/model-z" target="_blank">https://huggingface.co/datasets/mikeee/model-z</a>
    the first this app is run. This needs to be done
    only **once** since the files are cached locally.

    The app can be run offline afterwards.
# """

about = pn.pane.Markdown(
    about_info,
    style={
        "background-color": "#F6F6F6",
        "border": "2px solid black",
        "border-radius": "5px",
        "padding": "14px",
        "font-size": "medium",
        "color": "#10874a",  # "coral",
    },
)
# "font-size": "18px",


def filevalue2list(value: bytes) -> List[str]:
    """Convert file1.value to list."""
    if value is None:
        return []

    if not isinstance(value, bytes):
        raise Exception("not bytes fed to me, cant handle it.")

    encoding = chardet.detect(value).get("encoding") or "utf8"
    try:
        _ = value.decode(encoding=encoding)
    except Exception as e:
        logger.error(e)
        raise
    return [elm.strip() for elm in _.splitlines() if elm.strip()]


def gen_df():
    """Gen df (pandas.DataFrame) from file1.value/file2.value."""
    # global file1, file2

    list1 = filevalue2list(file1.value)
    list2 = filevalue2list(file2.value)

    if VIZBEE_DEV:
        # set default during dev
        if not len(list1):
            text1 = loadtext(Path(__file__).parent / "test-en.txt")
            list1 = [elm.strip() for elm in text1.splitlines() if elm.strip()]
        if not len(list2):
            text2 = loadtext(Path(__file__).parent / "test-zh.txt")
            list2 = [_.strip() for _ in text2.splitlines() if _.strip()]

    ns.df = pd.DataFrame(
        zip_longest(list1, list2, [], fillvalue=""),
        columns=["text1", "text2", "metric"],
    )

    ns.df.insert(0, "seq", range(1, 1 + len(ns.df)))
    ns.df = ns.df.set_index("seq")

    return ns.df


def cb_submit(event=param.parameterized.Event):
    """Callback for the submit button (upload)."""
    # global df  # , df_pane
    ns.counter += 1
    # ic(" cb_submit ", ns.counter)

    logger.debug("ns.counter: %s", ns.counter)

    # update df_pane
    ns.df = gen_df()

    # df_pane = pn.pane.DataFrame(df, width=400)
    df_pane = pn.pane.DataFrame(
        ns.df,
        justify="left",
        sizing_mode="stretch_width",
    )

    # update tab1
    s_cb_load()


def s_cb_load(event=param.parameterized.Event):
    """Callback1 for tab1 (submit)."""
    # global df, df_pane

    ns.counter += 1
    # ic("cb_submit", ns.counter)

    logger.debug("ns.counter: %s", ns.counter)

    # template.main[0].objects = [pn.Column("# Page 1", f"modi..bla bla bla  {ns.counter}")]

    # df = gen_df()

    # ns.df_pane = pn.pane.DataFrame(df, width=400)
    ns.df_pane = pn.pane.DataFrame(
        ns.df,
        justify="left",
        sizing_mode="stretch_width",
    )

    tab1 = pn.Column(
        pn.Row(file1, file2, button_submit),
        ns.df_pane,
    )

    template.main[0].objects = [tab1]

    return "Page 1"


editors = {
    # "text1": StringEditor(),
    # "seq": {"type": "number", "min": 1, "step": 1},
    "text1": TextEditor(),
    "text2": TextEditor(),
    "metric": {"type": "number", "min": -1, "max": 1, "step": 0.1},
    # 'bool': {'type': 'tickCross', 'tristate': True},
    # 'str': TextEditor(),
    # 'str': StringEditor(),
    # 'str': {'type': 'string','values': True},
}
formatters = {
    # "seq": {},
    "text1": {"type": "textarea"},
    "text2": {"type": "textarea"},
    "metric": {"type": "progress", "max": 1, "min": 0},
}


def s_cb_align(event=param.parameterized.Event):
    """Callback2 for align tab."""
    # ic("cb 2 s_cb_align")

    logger.debug(" s_cb_align")

    ns.counter += 1
    # template.main[0].objects = [pn.Column("# Page 2", f"modi..bla bla bla {ns.counter}")]

    tab2 = pn.Column(
        pn.Row(file1, file2),
        ns.df_pane,
    )

    edit_table = pn.widgets.Tabulator(
        ns.df,
        editors=editors,
        formatters=formatters,
        pagination="remote",
        # page_size=50,
        # max_height=60,
        row_height=60,
        max_width=1500,
        # loading=True,
        widths={"text1": 400, "text2": 400, "metric": 100},
    )
    tab2 = pn.Column(
        edit_table
    )
    template.main[0].objects = [tab2]

    return "Page 2"


@unsync
def s_cb_about(event):
    """Callback for about."""
    ns.counter += 1
    # ic("cb3 ", ns.counter)

    logger.debug(" ns.counter: %s", ns.counter)

    template.open_modal()
    time.sleep(10)

    # ic("cb3 done")
    template.close_modal()


button_submit = pn.widgets.Button(name="Submit")
button_submit.on_click(cb_submit)

s_btn_load.on_click(s_cb_load)
s_btn_align.on_click(s_cb_align)
s_btn_about.on_click(s_cb_about)  # modi

page = pn.Column(sizing_mode="stretch_width")

# ishow = pn.bind(show, page=page)
# pn.state.location.sync(page, {"value": "page"})

ACCENT_COLOR = "#0072B5"
# ACCENT_COLOR = "#0474BB"
# ACCENT_COLOR = "#0878BF"
# ACCENT_COLOR = "#FF78BF"
DEFAULT_PARAMS = {
    # "site": "Panel Multi Page App",
    "site": "Align with Style",
    "title": "Visbee",
    "accent_base_color": ACCENT_COLOR,
    "header_background": ACCENT_COLOR,
}
template = pn.template.FastListTemplate(
    sidebar=[s_btn_load, s_btn_align, s_btn_about],  # modi
    # main=[ishow],
    # main=[page],
    # sidebar=[btn_submit, btn_align],
    # main=[pages['Page 1'],
    **DEFAULT_PARAMS
)

template.main.append(page)
template.modal.append(about)

s_cb_load()  # set inital page 1

dashboard = template

if __name__.startswith("bokeh"):
    # main().servable(title="Vizbee Aligner")
    dashboard.servable(title="Vizbee")

if __name__ == "__main__":
    pn.serve(
        # main(),
        dashboard,
        title="Vizbee",
        port=8088,
        verbose=True,
    )

_ = """
pn.serve(
    panels,
    port=0,
    address=None,
    websocket_origin=None,
    loop=None,
    show=True,
    start=True,
    title=None,
    verbose=True,
    location=True,
    threaded=False,
    **kwargs,
)
# """
