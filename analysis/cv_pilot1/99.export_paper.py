#%% imports and definition
import os

from routine.svgutils.compose import Figure
from routine.utilities import make_svg_panel, svg_unique_id

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./figs/paper"
os.makedirs(OUT_PATH, exist_ok=True)

#%% make fig2
w_gap = 10
h_gap = 0
sh_left = (0, 0)
svgs = {
    "A": "./figs/external/linear_track.svg",
    "B": "./figs/frame_label/example.svg",
    "C": "./figs/external/confocal.svg",
    "D": "./figs/register_g2r/cells/m15.svg",
    "E": "./figs/register_g2r/traces.svg",
}
for fn in svgs.values():
    svg_unique_id(fn)

panA = make_svg_panel("A", svgs["A"], PARAMT_TEXT, im_scale=0.6, sh=sh_left)
panB = make_svg_panel("B", svgs["B"], PARAMT_TEXT, sh=sh_left)
panC = make_svg_panel("C", svgs["C"], PARAMT_TEXT, sh=(0, 3))
panD = make_svg_panel("D", svgs["D"], PARAMT_TEXT, sh=sh_left)
panE = make_svg_panel("E", svgs["E"], PARAMT_TEXT, sh=sh_left)

h_row1 = max(panA.height, panB.height)
h_row2 = max(panC.height, panD.height)
h_row3 = panE.height
w_col1 = max(panA.width, panC.width)
w_col2 = max(panB.width, panD.width)
h_fig = h_row1 + h_row2 + h_row3 + 2 * h_gap
w_fig = max(w_col1 + w_col2 + w_gap, panE.width)
fig = Figure(
    w_fig,
    h_fig,
    panA,
    panB.move(x=w_col1 + w_gap, y=0),
    panC.move(x=0, y=h_row1 + h_gap),
    panD.move(x=w_col1 + w_gap, y=h_row1 + h_gap),
    panE.move(x=(w_fig - panE.width) / 2, y=h_row1 + h_row2 + 2 * h_gap),
)
fig.save(os.path.join(OUT_PATH, "fig2.svg"))

#%% make fig3
h_gap = 0
w_gap = 20
sh_left = (0, 0)
svgs = {
    "A": "./figs/register_g2r/summary_agg.svg",
    "B": "./figs/drift/drifting_cells.svg",
    "C": "./figs/drift/overlap-actMean.svg",
    "D": "./figs/drift/pv_corr-by_cell_med.svg",
}
for fn in svgs.values():
    svg_unique_id(fn)

panA = make_svg_panel("A", svgs["A"], PARAMT_TEXT, sh=sh_left)
panB = make_svg_panel("B", svgs["B"], PARAMT_TEXT, sh=sh_left)
panC = make_svg_panel("C", svgs["C"], PARAMT_TEXT, sh=sh_left)
panD = make_svg_panel("D", svgs["D"], PARAMT_TEXT, sh=sh_left)

h_row1 = panA.height
h_row2 = panB.height
h_row3 = max(panC.height, panD.height)
w_col1 = panC.width
w_col2 = panD.width
h_fig = h_row1 + h_row2 + h_row3 + 2 * h_gap
w_fig = max(w_col1 + w_col2 + w_gap, panA.width, panB.width)
pad_row3 = (w_fig - w_col1 - w_gap - w_col2) / 2
fig = Figure(
    w_fig,
    h_fig,
    panA.move(x=(w_fig - panA.width) / 2, y=0),
    panB.move(x=(w_fig - panB.width) / 2, y=h_row1 + h_gap),
    panC.move(x=pad_row3, y=h_row1 + h_row2 + 2 * h_gap),
    panD.move(x=pad_row3 + w_col1 + w_gap, y=h_row1 + h_row2 + 2 * h_gap),
)
fig.save(os.path.join(OUT_PATH, "fig3.svg"))

#%% make S1
h_gap = 0
w_gap = 20
svgs = {
    "A": "./figs/hub_cells/peak_cdf.svg",
    "B": "./figs/hub_cells/metrics.svg",
}
for fn in svgs.values():
    svg_unique_id(fn)

panA = make_svg_panel("A", svgs["A"], PARAMT_TEXT, sh=sh_left)
panB = make_svg_panel("B", svgs["B"], PARAMT_TEXT, sh=sh_left)

h_fig = panA.height + h_gap + panB.height
w_fig = max(panA.width, panB.width)
fig = Figure(
    w_fig,
    h_fig,
    panA.move(x=(w_fig - panA.width) / 2, y=0),
    panB.move(x=(w_fig - panB.width) / 2, y=panA.height + h_gap),
)
fig.save(os.path.join(OUT_PATH, "figS1.svg"))
