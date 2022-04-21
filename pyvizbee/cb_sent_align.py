"""Define callback to sent-align."""
import param
import panel as pn
import logzero
from logzero import logger

from pyvizbee.ns import ns
from pyvizbee.template import s_cb_align
from pyvizbee.sent_align_ns_df import sent_align_ns_df
from pyvizbee.loglevel import loglevel


def cb_sent_align(event=param.parameterized.Event):
    """Define callback to sent-align (in tab3): button_sent_align.on_click(cb_sent_align)."""
    logger.debug("cb_sent_align")

    # process ns.df: align_ns_df
    try:
        # ns.df = align_ns_df(ns.df, ns.model)
        ns.df = sent_align_ns_df(ns.df, lang1=ns.lang1, lang2=ns.lang2)
    except Exception:
        logger.exception("sent_align_ns_df(ns.df, lang1, lang2)")
        logger.warning("We continue and hope for the best.")

    logzero.loglevel(loglevel())
    logger.debug(" len(ns.df): %s", len(ns.df))
    logger.debug(" ns.df[:10]: \n%s", ns.df[:10])

    # update tab3
    logger.debug("update tab3: s_cb_align()")

    s_cb_align()


button_sent_align = pn.widgets.Button(name="Sent-Align")
button_sent_align.on_click(cb_sent_align)
