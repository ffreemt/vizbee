"""vizbee based on holoviz panel.

panel
    pn.pane: markdown
    pn.pane.HTML

    interact  @interact: function with str or number input
    bind
    @depends

tab1 tab2 tab3
    tab1 load files
    tab2
    tab3
    tab3 help info

pn.layout.Tabs(('Intro', str_pane), ('Table', tabulator_pane), ('HTML', html_pane)).servable()

"""
# pylint: disable=invalid-name, global-statement, unused-argument

from itertools import zip_longest

# import io
# import logging
# from pathlib import Path
from textwrap import dedent
from typing import List

import cchardet as chardet
import environs
import logzero

# import numpy as np
import pandas as pd
import panel as pn
import param
from logzero import logger

# from pyvizbee.load_text_widget import load_text_widget
# from pyvizbee.display_text_widget import display_text_widget

# set env LOGLEVEL to turn on 10/debug/DEBUG to turn on debug
try:
    _ = environs.Env().log_level("LOGLEVEL")
# except environs.EnvValidationError:
except (environs.EnvError, environs.EnvValidationError):
    _ = None
except Exception:
    _ = None
logzero.loglevel(_ or 10)

logger.debug(" debug: %s", __file__)
logger.info(" info: %s", __file__)

pn.config.sizing_mode = "stretch_both"
pn.config.sizing_mode = "stretch_width"
pn.extension("tabulator")
file1: pn.widgets.input.FileInput = None
file2: pn.widgets.input.FileInput = None
tab1: pn.layout.Tabs = None
tab2: pn.layout.Tabs = None
df = pd.DataFrame


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

    return pd.DataFrame(
        zip_longest(list1, list2, [], fillvalue=""),
        columns=["text1", "text2", "likelihood"],
    )


def load_text_widget(event=param.parameterized.Event):
    """Ready tab1."""
    global file1, file2
    file1 = pn.widgets.FileInput(accept=".txt,.csv,.tsv")
    file2 = pn.widgets.FileInput(accept=".txt,.csv,.tsv")
    return pn.Column(
        # "Load text",
        # pn.Row(file1, file2),
        # sizing_mode="fixed",
        file1,
        file2,
    )


def display_text_widget(event=param.parameterized.Event):
    """Ready tab2."""
    global df

    logger.debug(" file1 is None: %s", file1 is None)

    if hasattr(file1, "value") and file1.value is not None:
        logger.debug(" file1.value is not None: %s", file1.value is not None)

    if file1 is not None and file1.value is not None:
        logger.debug(" file1.value[:10]: %s", file1.value[:10])
        logger.debug(" file1.filename: %s", file1.filename)

        _ = """
        # df = pd.read_csv(io.BytesIO(file1.value), header=None, delimiter="\n", names=[file1.filename])
        encoding = chardet.detect(file1.value).get("encoding")
        if encoding is None:  # fallback to utf8
            encoding = "utf8"
        _ = file1.value.decode(encoding)

        # remove blank lines
        _ = [elm.strip() for elm in _.splitlines() if elm.strip()]
        df = pd.DataFrame(_)
        # """

        df = gen_df()

        # df_pane = pn.pane.DataFrame(df, width=400)
        df_pane = pn.pane.DataFrame(df, width=400)

        logger.debug("df.head(3): %s", df.head(3))

        tab2_obj = pn.Column(
            pn.Row(file1, file2),
            df_pane,
        )

        # update
        if tab2 is not None:
            logger.debug(" update tab2")
            dashboard[1] = ("Content", tab2_obj)
            return tab2_obj

        return tab2_obj

    # if "file1" in global() and "file2" in gloabls():
    # if {"file1", "file2"} <= globals().keys():

    return pn.Column(
        # pn.Row(file1, file2),
        file1,
        file2,
    )


_ = '''
def main() -> pn.layout.Tabs:
    """Prepare servable tabs."""
    global tab2
# '''
# if 1:

about_info = dedent(
    """
    ## Visbee Aligner

    from mu's keyborad in cyberspace (join qq group **316287378** to be kept informed and for questions and feedback)

    <hr>

    Two large model files (1.2G and 500M) will be downloaded (from
    [https://huggingface.co/datasets/mikeee/model-z](https://huggingface.co/datasets/mikeee/model-z))
    the first this app is run. This needs to be done
    only **once** since the files are cached locally.

    The app can be run offline afterwards.
    """
).strip()

about = pn.pane.Markdown(
    about_info,
    style={
        "background-color": "#F6F6F6",
        "border": "2px solid black",
        "border-radius": "5px",
        "padding": "14px",
        # "font-size": "18px",
        "font-size": "medium",
        "color": "#10874a",  # "coral",
    },
)
# about = pn.Column(about_info)
# about = pn.pane.Alert(about_info, alert_type="success")

load_w = load_text_widget()
# content_w = display_text_widget()

# Tab 1
content_w = pn.param.ParamFunction(
    display_text_widget,
    lazy=True,
    name="Content",
)
button_submit = pn.widgets.Button(name="Submit")
button_submit.on_click(display_text_widget)

tab1 = pn.Column(
    content_w,
    button_submit,
    name="Load files",
    sizing_mode=None,
)

# p2 = pn.param.ParamFunction(plot, lazy=True, name='Seed 1')
tab2 = pn.param.ParamFunction(display_text_widget, lazy=True, name="Content")

_ = '''
file1_content = ""
logger.debug("Evaluate file1.value...")
if "file1" in globals():
    if file1.value:  # type: ignore
        file1_content = io.StringIO(file1.value).read()

debug = dedent(f"""
    file1 content: [{file1_content}]
""").strip()
# '''

dashboard = pn.layout.Tabs(
    # ("Load files", load_w),
    tab1,
    tab2,
    ("About", about),
    dynamic=True,
    name="Vizbee",
)
# return dashboard

if __name__.startswith("bokeh"):
    # main().servable(title="Vizbee Aligner")
    dashboard.servable(title="Vizbee Aligner")

if __name__ == "__main__":
    pn.serve(
        # main(),
        dashboard,
        title="Vizbee Aligner",
        port=8088,
        verbose=True,
    )
