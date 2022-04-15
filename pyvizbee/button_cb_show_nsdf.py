"""Define button_show_nsdf cb_show_nsdf."""
import panel as pn
import param
from logzero import logger

from pyvizbee.ns import ns


def cb_show_nsdf(event=param.parameterized.Event):
    """Callback to cb_show_nsdf (in tab3)."""
    logger.debug("cb_show_nsdf")
    ...
    logger.debug("ns.df: %s", ns.df)


button_show_nsdf = pn.widgets.Button(name="ShowNsdf")
button_show_nsdf.on_click(cb_show_nsdf)
