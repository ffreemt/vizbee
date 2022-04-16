"""Set up template."""
# pylint: disable=unused-argument, unused-import, too-many-locals, too-many-statements

import os
import time
from itertools import zip_longest
from pathlib import Path
# from pprint import pprint
from textwrap import dedent
# from types import SimpleNamespace
from typing import List

import cchardet as chardet
import logzero
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models.widgets.tables import TextEditor
from cmat2aset import cmat2aset
from fast_scores.gen_cmat import gen_cmat
from fast_scores.en2zh import en2zh
from fast_scores.process_en import process_en
from fast_scores.gen_model import gen_model
from logzero import logger
from unsync import unsync

from fastlid import fastlid

from pyvizbee import __version__
from pyvizbee.loadtext import loadtext
from pyvizbee.loglevel import loglevel

from pyvizbee.align_ns_df import align_ns_df
from pyvizbee.ns import ns, default_eps, default_min_samples

from pyvizbee.cb_save_xlsx import cb_save_xlsx
from pyvizbee.cb_save_tsv import cb_save_tsv
from pyvizbee.cb_show_nsdf import cb_show_nsdf

VIZBEE_DEV = os.environ.get("VIZBEE_DEV")

_ = r"""
# better set PYTHONPATH=..\fast-scores;..\cmat2aset
if VIZBEE_DEV:
    # from icecream import ic
    from add_path import add_path

    add_path(
        [
            r"..\cmat2aset",
            r"..\fast-scores",
        ]
    )
    try:
        logger.debug(" importing fast_scores")
        # import e zbee
        # import fast_scores
        from fast_scores.gen_cmat import gen_cmat  # noqa
    except Exception:
        logger.exception("import fast_scores errors")

from cmat2aset import cmat2aset  # noqa
from fast_scores.gen_cmat import gen_cmat  # noqa
# """


def filevalue2list(value: bytes) -> List[str]:
    """Convert file1.value to list."""
    if value is None:
        return []

    if not isinstance(value, bytes):
        raise Exception("not bytes fed to me, cant handle it.")

    encoding = chardet.detect(value).get("encoding") or "utf8"
    try:
        _ = value.decode(encoding=encoding)
    except Exception as _:
        logger.error(_)
        raise
    return [elm.strip() for elm in _.splitlines() if elm.strip()]


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


# callbacks for tab buttons
def cb_submit(event=param.parameterized.Event):
    """Callback for the submit button (upload)."""
    # global df  # , df_pane
    ns.counter += 1

    logger.debug("ns.counter: %s", ns.counter)

    # update df_pane
    ns.df = gen_df()

    # df_pane = pn.pane.DataFrame(df, width=400)
    df_pane = pn.pane.DataFrame(
        ns.df,
        justify="left",
        sizing_mode="stretch_width",
    )
    del df_pane

    # update tab1
    s_cb_load()


def cb_reset(event=param.parameterized.Event):
    """Callback for the reset button (eps/min_samples)."""
    logger.debug("cb_reset")
    logger.info("cb_reset")
    ns.eps = default_eps
    ns.min_samples = default_min_samples
    s_cb_params()


def cb_toggle_params(event=param.parameterized.Event):
    """Callback to toggle params tab."""
    if "off" in s_btn_params.name:
        s_btn_params.name = "Plot (on)"
    else:
        s_btn_params.name = "Plot (off)"


def cb_align(event=param.parameterized.Event):
    """Callback to align (in tab3): button_align.on_click(cb_align)."""
    logger.debug("cb_align")

    # process ns.df: align_ns_df
    try:
        ns.df = align_ns_df(ns.df, ns.model)
        # ns.df = align_ns_df(ns.df)
    except Exception:
        logger.exception("align_ns_df(ns.df, ns.model)")
        logger.warning("We continue and hope for the best.")

    logzero.loglevel(loglevel())
    logger.debug(" len(ns.df): %s", len(ns.df))
    logger.debug(" ns.df[:10]: \n%s", ns.df[:10])

    # update tab3
    logger.debug("update tab3: s_cb_align()")
    s_cb_align()


# click buttons in tabs
button_submit = pn.widgets.Button(name="Submit")
button_submit.on_click(cb_submit)

button_reset = pn.widgets.Button(name="Reset")
button_reset.on_click(cb_reset)

button_toggle_params = pn.widgets.Button(name="TogglePlot")
button_toggle_params.on_click(cb_toggle_params)

button_align = pn.widgets.Button(name="Align/Re-align")
button_align.on_click(cb_align)

button_show_nsdf = pn.widgets.Button(name="ShowNsdf")
button_show_nsdf.on_click(cb_show_nsdf)

# download buttons in # tab3


# sidebar callbacks  # tab1
def s_cb_load(event=param.parameterized.Event):
    """Callback1 for tab1 (submit) tab1."""
    # global df, df_pane

    ns.counter += 1

    logger.debug("ns.counter: %s", ns.counter)

    # template.main[0].objects = [pn.Column("# Page 1", f"modi..bla bla bla  {ns.counter}")]

    # df = gen_df()

    # ns.df_pane = pn.pane.DataFrame(df, width=400)
    ns.df_pane = pn.pane.DataFrame(
        ns.df,
        justify="left",
        sizing_mode="stretch_width",
    )

    tab = pn.Column(
        pn.Row(
            ns.file1,
            ns.file2,
            pn.Column(button_submit, button_toggle_params),
        ),
        ns.df_pane,
    )

    template.main[0].objects = [tab]

    return "Page 1"


# tab2
@unsync
def s_cb_params(event=param.parameterized.Event):
    """Callback for (params) tab2.

    int_slider = pn.widgets.IntSlider(name='Integer Slider', start=0, end=8, step=2, value=4)

    int_slider

    https://www.quansight.com/post/panel-holoviews-learning-aid
    def update(event):
        if event.obj is radio_button:
            radio_button_display.object = f'Radio Button Value: {radio_button.value}'
    radio_button.param.watch(update, "value")
    """
    try:
        import holoviews as hv  # pylint: disable=W,C,R
        pn.extension("plotly")
        hv.extension("plotly")
        flag = False
    except ModuleNotFoundError:
        flag = True
    except Exception as exc:
        logger.exception(exc)
        raise SystemExit(1) from exc

    # if 1:
    if flag:
        _ = dedent(
            f"""You need to install holoviews, e.g.,
                pip install holoviews or pip install
                pyvizbee[plot]=={__version__}"""
        ).strip()
        tab = pn.Column(_)
        template.main[0].objects = [tab]
        return

    # to silence pyright
    import holoviews as hv  # pylint: disable=W,C,R

    if "off" in s_btn_params.name:
        tab = pn.Column("Click TogglePlot in the Load Files tab if you want to see plots...Plotting can take a long time and computer resources (CPU, RAM etc) for large text bodies.")

        # https://panel.holoviz.org/reference/panes/Alert.html
        # primary, secondary, success, danger, warning, info, light, dark
        # alert = pn.pane.Alert("success! ", alert_type="success")
        alert = pn.pane.Alert("Plotting is off! ", alert_type="warning")

        template.main[0].objects = [alert]
        time.sleep(3)
        template.main[0].objects = [tab]
        return

    if ns.df is None:
        tab = pn.Column(" Load files first...")
        template.main[0].objects = [tab]
        return

    def update(event):
        if event.obj is w_eps:
            logger.debug("w_eps.value: %s", w_eps.value)
            ns.eps = w_eps.value
        if event.obj is w_min_samples:
            logger.debug("w_min_samples.value: %s", w_min_samples.value)
            ns.min_samples = w_min_samples.value

        # update self to update headmaps
        s_cb_params()

    logger.debug(" s_cb_params ")

    w_eps = pn.widgets.FloatSlider(
        name="epsilon", start=0, end=20, step=0.2, value=ns.eps
    )
    w_min_samples = pn.widgets.IntSlider(
        name="min samples", start=1, end=20, step=1, value=ns.min_samples
    )

    w_eps.param.watch(update, "value")
    w_min_samples.param.watch(update, "value")

    xlabel = "x"
    ylabel = "y"

    _ = [elm for elm in ns.df.text1.to_list() if elm.strip()]
    logger.debug("first[:10]: %s", _[:10])
    logger.debug("second[:10]: %s", ns.df.text2.to_list()[:10])

    if not (_ and ns.df.text2.to_list()):
        _ = "One or both inputs empty... nothing to do"
        alert = pn.pane.Alert(_, alert_type="warning")
        template.main[0].objects = [alert]
        return

    try:
        cmat = gen_cmat(_, ns.df.text2.to_list())
    except Exception as exc:
        logger.exception("gen_cmat")
        tab = str(exc)
        template.main[0].objects = [tab]
        return

    # restore
    logzero.loglevel(loglevel())

    _ = [[*elm, v] for elm, v in np.ndenumerate(cmat)]
    hm1 = hv.HeatMap(
        _,
        label="(z) likelihood heatmap",
        name="name1",
    )
    hm1.opts(
        # xticks=None,
        # tools=['hover'],  # only for bokeh
        # colorbar=True,
        # cmap="viridis_r",
        xlabel=xlabel,
        ylabel=ylabel,
        # cmap="viridis_r",
        # cmap="viridis",
        cmap="gist_earth_r",
        # cmap="summer_r",
        # cmap="fire_r",
    )

    aset = cmat2aset(
        cmat,
        eps=ns.eps,
        min_samples=ns.min_samples,
    )

    logzero.loglevel(loglevel())  # fastlid reset logzero.loglevel(20)
    logger.debug("esp: %s, min-samples: %s", ns.eps, ns.min_samples)
    logger.debug("aset: \n%s", aset)

    arr = np.array(aset, dtype=object)

    # convert "" in col3 to nan
    arr[:, 2][arr[:, 2] == ""] = np.nan
    arr[arr == ""] = 0
    arr = arr.astype(float)

    # set value to np.nan for "": old way
    # _ = [elm.tolist() if elm[2] != 0 else elm.tolist()[:2]
    # + [np.nan] for elm in arr]

    hm2 = hv.HeatMap(
        arr,
        label="(z) likelihood align trace",
    )

    hm2.opts(
        # xticks=None,
        # tools=['hover'],  # only for bokeh
        colorbar=True,
        # cmap="viridis_r",
        xlabel=xlabel,
        ylabel=ylabel,
        # cmap="viridis_r",
        # cmap="viridis",
        cmap="gist_earth_r",
        # cmap="summer_r",
        # cmap="fire_r",
        # labelled=["en", "zh", "llh"]  # ["x", "y", "z"]
    )

    tab = pn.Column(
        pn.Row(
            w_eps,
            w_min_samples,
            button_reset,
        ),
        pn.Row(hm1 + hm2),
    )
    template.main[0].objects = [tab]

    # return "Params tab"


editors = {
    # "text1": StringEditor(),
    # "seq": {"type": "number", "min": 1, "step": 1},
    "text1": TextEditor(),
    "text2": TextEditor(),
    "metric": {"type": "number", "min": -1, "max": 1, "step": 2},
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


# tab3
def s_cb_align(event=param.parameterized.Event):
    """Callback2 for align tab3."""
    #

    logger.debug(" s_cb_align")

    if ns.df is None:
        tab = pn.Column(" Load files first...")
        template.main[0].objects = [tab]
        return

    ns.counter += 1
    # template.main[0].objects = [pn.Column("# Page 2", f"modi..bla bla bla {ns.counter}")]
    edit_table = pn.widgets.Tabulator(
        value=ns.df,
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

    if ns.file1.filename:
        stem = Path(ns.file1.filename).stem + "-ali"
    else:
        stem = "aligned"

    # button_save_xlsx = pn.widgets.FileDownload(filename="temp.xlsx", callback=cb_save_xlsx, button_type="primary")
    button_save_xlsx = pn.widgets.FileDownload(filename=f"{stem}.xlsx", callback=cb_save_xlsx)
    button_save_tsv = pn.widgets.FileDownload(filename=f"{stem}.tsv", callback=cb_save_tsv)

    if VIZBEE_DEV:
        _ = pn.Row(button_align, button_save_xlsx, button_save_tsv, button_show_nsdf)
    else:  # no button_show_nsdf
        _ = pn.Row(button_align, button_save_xlsx, button_save_tsv)

    tab = pn.Column(
        _,
        pn.layout.HSpacer(),
        edit_table,
    )
    template.main[0].objects = [tab]


# tab4 modal
@unsync
def s_cb_about(event):
    """Callback for about tab4 (modal)."""
    ns.counter += 1

    _ = """
    # does not work
    if ns.counter % 2:
        template.modal.objects = [about]
        logger.debug(" ns.counter: %s, modal: %s", ns.counter, pprint(template.modal))
    else:
        _ = pn.pane.Markdown(
            " 111 ",
            style={
                "background-color": "#F6F6F6",
                "border": "2px solid black",
                "border-radius": "5px",
                "padding": "14px",
                "font-size": "medium",
                "color": "#10874a",  # "coral",
            },
        )
        template.modal.objects = [_]
        logger.debug(" ns.counter: %s, modal: %s", ns.counter, pprint(template.modal))
    # """

    template.open_modal()
    time.sleep(10)
    template.close_modal()


# modi sidebar button
s_btn_load = pn.widgets.Button(name="Load Files")
s_btn_params = pn.widgets.Button(name="Plot (off)")
s_btn_align = pn.widgets.Button(name="Align/Save")
s_btn_about = pn.widgets.Button(name="About")

# modi sidebar buttons
s_btn_load.on_click(s_cb_load)
s_btn_params.on_click(s_cb_params)
s_btn_align.on_click(s_cb_align)
s_btn_about.on_click(s_cb_about)  # modi

page = pn.Column(sizing_mode="stretch_width")

about_info = dedent(
    """
    ## Vizbee Aligner

    Greetings from mu's keyboard in cyberspace (join qq group **316287378** or visit the bumblebee forum [bumblebee.freeforums.net](https://bumblebee.freeforums.net/board/1/general-discussion) to be kept informed and for questions and feedback)
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

ACCENT_COLOR = "#0072B5"
# ACCENT_COLOR = "#0474BB"
# ACCENT_COLOR = "#0878BF"
# ACCENT_COLOR = "#FF78BF"
DEFAULT_PARAMS = {
    # "site": "Panel Multi Page App",
    "site": "Align with Style",
    "title": "Vizbee",
    "accent_base_color": ACCENT_COLOR,
    "header_background": ACCENT_COLOR,
}
template = pn.template.FastListTemplate(
    sidebar=[
        s_btn_load,
        s_btn_params,
        s_btn_align,
        s_btn_about,
    ],  # modi
    # main=[ishow],
    # main=[page],
    # sidebar=[btn_submit, btn_align],
    # main=[pages['Page 1'],
    **DEFAULT_PARAMS,
)

template.main.append(page)
template.modal.append(about)
