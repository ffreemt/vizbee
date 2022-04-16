"""Define button_save_tsv cb_save_tsv."""
import io
import pandas as pd
import param

from pyvizbee.ns import ns
from pyvizbee.color_map import color_map


def cb_save_xlsx(event=param.parameterized.Event):
    """Callback to button_save_xlsx (in # tab3).

    https://panel.holoviz.org/reference/widgets/FileDownload.html

    button_type (str): A button theme; should be one of 'default' (white), 'primary' (blue), 'success' (green), 'info' (yellow), or 'danger' (red)

    from bokeh.sampledata.autompg import autompg

    from io import StringIO
    sio = StringIO()
    autompg.to_csv(sio)
    sio.seek(0)

    pn.widgets.FileDownload(sio, embed=True, filename='autompg.csv')

    https://pyviz-dev.github.io/panel/gallery/simple/file_download_examples.html
    """
    subset = list(ns.df.columns[2:3])  # 3rd col
    s_df = ns.df.style.applymap(color_map, subset=subset)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:  # pylint: disable=abstract-class-instantiated
        s_df.to_excel(writer, index=False, header=False, sheet_name="Sheet1")
        writer.sheets["Sheet1"].set_column("A:A", 70)
        writer.sheets["Sheet1"].set_column("B:B", 70)

    output.seek(0)

    return output
