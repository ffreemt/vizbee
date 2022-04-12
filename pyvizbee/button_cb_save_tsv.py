"""Define button_save_tsv cb_save_tsv."""
import panel as pn
import param
from logzero import logger

# from pyvizbee.__main__ import template  # circular import
# from .__main__ import template


def cb_save_tsv(event=param.parameterized.Event):
    """Callback to save_tsv (in tab3)."""
    logger.debug("cb_save_tsv")
    ...


button_save_tsv = pn.widgets.Button(name="SaveTsv")
button_save_tsv.on_click(cb_save_tsv)
