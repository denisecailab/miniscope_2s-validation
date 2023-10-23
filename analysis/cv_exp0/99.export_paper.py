# %% imports and definition
import os

from routine.svgutils.compose import Figure
from routine.utilities import make_svg_panel, svg_unique_id

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./figs/paper"
os.makedirs(OUT_PATH, exist_ok=True)

# %% make experiment figure
w_gap = 10
h_gap = 25
sh_left = (0, 0)
sh_top = (0, 14)
svgs = {
    "A": "./figs/external/linear_track.svg",
    "B": "./figs/behav_comparison/example.svg",
    "C": "./figs/behav_comparison/comparison.svg",
}
for fn in svgs.values():
    svg_unique_id(fn)

panA = make_svg_panel("A", svgs["A"], PARAMT_TEXT, im_scale=0.065, sh=sh_top)
panB = make_svg_panel("B", svgs["B"], PARAMT_TEXT, sh=sh_left)
panC = make_svg_panel("C", svgs["C"], PARAMT_TEXT, sh=sh_top)

h_col1 = panA.height + panC.height + h_gap
w_col1 = max(panA.width, panC.width)
h_fig = max(h_col1, panB.height)
w_fig = w_col1 + w_gap + panB.width
fig = Figure(
    w_fig,
    h_fig,
    panA,
    panB.move(x=w_col1 + w_gap, y=0),
    panC.move(x=0, y=panA.height + h_gap),
)
fig.save(os.path.join(OUT_PATH, "2s_experiment.svg"))

# %% make drift figure
h_gap = 1
sh_left = (0, 0)
svgs = {
    "A": "./figs/drift/overlap-actMean.svg",
    "B": "./figs/drift/pv_corr-place_cells.svg",
}
for fn in svgs.values():
    svg_unique_id(fn)

panA = make_svg_panel("A", svgs["A"], PARAMT_TEXT, sh=sh_left)
panB = make_svg_panel("B", svgs["B"], PARAMT_TEXT, sh=sh_left)

h_fig = panA.height + panB.height + h_gap
w_fig = max(panA.width, panB.width)
fig = Figure(w_fig, h_fig, panA, panB.move(x=0, y=panA.height + h_gap))
fig.save(os.path.join(OUT_PATH, "drift.svg"))

# %% make S1
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
