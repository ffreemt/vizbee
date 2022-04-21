"""vizbee based on holoviz panel.

set VIZBEE_DEV=1
from add_path import add_path

sidebars
s_btn_ load params align about

callbacks to display tabs, s_btn_x.on_click(s_cb_x)
s_cb load params align about

buttons in tabs for
button_ reset(eps, min_samples slides), submit(load files), toggle_params, button_align

callbacks associated with buttons above
cb_reset, cb_submit, cb_toggle_params
cb_align


"""
# pylint: disable=invalid-name, unused-argument, unused-import, wrong-import-position, wrong-import-order, too-many-locals, too-many-statements

import logzero
import panel as pn
from logzero import logger
from pyvizbee.loglevel import loglevel

from pyvizbee.template import template, s_cb_load

logzero.loglevel(loglevel())

logger.debug(" debug: %s", __file__)
logger.info(" info: %s", __file__)

pn.config.sizing_mode = "stretch_width"
pn.extension("tabulator")

s_cb_load()  # set inital page 1

if __name__.startswith("bokeh"):
    # main().servable(title="Vizbee Aligner")
    template.servable(title="Vizbee")

if __name__ == "__main__":
    pn.serve(
        # main(),
        template,
        title="Vizbee",
        port=5006,
        # address="0.0.0.0",  # to allow external-ip
        # linux: export BOKEH_ALLOW_WS_ORIGIN=*
        # win: set BOKEH_ALLOW_WS_ORIGIN=*
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
