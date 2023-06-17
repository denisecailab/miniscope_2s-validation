# %% imports and definition
import base64
import os
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.image import imread
from PIL import Image
from routine.plotting import facet_plotly

IN_SS_CSV = "./log/sessions.csv"
IN_SNAPSHOTS = "./recording_snapshots"
FIG_PATH = "./figs/fov/"

os.makedirs(FIG_PATH, exist_ok=True)

# %% load images
ss_csv = (
    pd.read_csv(IN_SS_CSV)[["animal", "date", "session", "type"]]
    .astype({"date": str})
    .drop_duplicates()
)
for root, dirs, files in os.walk(IN_SNAPSHOTS):
    pngs_files = list(filter(lambda f: f.lower().endswith(".png"), files))
    if not pngs_files:
        continue
    anm = os.path.relpath(root, IN_SNAPSHOTS)
    png_data = []
    for png_f in pngs_files:
        snap_arr = imread(os.path.join(root, png_f))
        dt = png_f.split(".")[0]
        png_data.append(
            pd.Series(
                {
                    "animal": anm,
                    "date": dt,
                    "im_data": (snap_arr[:, :, :3] * 255).astype(np.uint8),
                }
            )
        )
    png_data = (
        pd.concat(png_data, axis="columns")
        .T.sort_values("date")
        .merge(ss_csv, on=["animal", "date"], how="left")
    )
    png_data = png_data[
        (png_data["type"] == "recording") | (png_data["date"] == "ref")
    ].copy()
    if not len(png_data) > 0:
        continue
    im_size = min([d.shape[0] for d in png_data["im_data"]])
    fig, layout = facet_plotly(
        png_data,
        facet_col="date",
        col_wrap=5,
        shared_xaxes="all",
        shared_yaxes="all",
        horizontal_spacing=0.01,
        vertical_spacing=0.04,
    )
    png_data = png_data.set_index("date")
    for _, ly in layout.reset_index().iterrows():
        r, c, dt = ly["row"], ly["col"], ly["col_label"]
        im_arr = png_data.loc[dt]["im_data"][:im_size, :im_size, :]
        im = Image.fromarray(im_arr)
        with BytesIO() as stream:
            im.save(stream, format="png")
            b64 = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(
                "utf-8"
            )
        fig.add_trace(
            go.Image(source=b64),
            row=r + 1,
            col=c + 1,
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(anm)))
