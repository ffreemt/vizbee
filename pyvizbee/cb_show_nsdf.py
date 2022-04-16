"""Define button_show_nsdf cb_show_nsdf."""
import param
from logzero import logger

from pyvizbee.ns import ns


def cb_show_nsdf(event=param.parameterized.Event):
    """Callback to cb_show_nsdf (in tab3)."""
    logger.debug("cb_show_nsdf")
    ...
    logger.debug("ns.df: %s", ns.df)
