"""Define button_save_tsv cb_save_tsv."""
import io
import param
from logzero import logger

from pyvizbee.ns import ns


def cb_save_tsv(event=param.parameterized.Event):
    """Callback to save_tsv (in # tab3)."""
    logger.debug("cb_save_tsv")
    ...
    output = io.BytesIO()
    ns.df.to_csv(output, sep="\t", index=False, header=False, encoding="gbk")

    output.seek(0)

    return output
