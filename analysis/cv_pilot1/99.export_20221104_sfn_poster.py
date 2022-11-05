#%% imports and definition
import os

from svgutils.compose import SVG, Figure, Panel, Text

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./figs/20221104_sfn_poster"
os.makedirs(OUT_PATH, exist_ok=True)


def make_panel(label, im_path, im_scale=1):
    im = SVG(im_path, fix_mpl=True).scale(im_scale)
    lab = Text(label, **PARAMT_TEXT)
    tsize = PARAMT_TEXT["size"]
    x_sh, y_sh = 0.6 * tsize, 1 * tsize
    pan = Panel(im.move(x=x_sh, y=y_sh), lab.move(x=0, y=y_sh))
    pan.height = im.height * im_scale + y_sh
    pan.width = im.width * im_scale + x_sh
    return pan


#%% make fig1
w_gap = 30
h_gap = 5
panA = make_panel("A", "./figs/external/linear_track.svg", 0.55)
panB = make_panel("B", "./figs/external/confocal.svg", 0.9)
panC = make_panel("C", "./figs/frame_label/example.svg")
panD = make_panel("D", "./figs/register_g2r/cells/m15.svg")
w_AB = max(panA.width, panB.width)
h_upper = max(panA.height + panB.height + h_gap, panC.height)
fig_h = h_upper + h_gap + panD.height
fig_w = max(w_AB + w_gap + panC.width, panD.width)
fig = Figure(
    fig_w,
    fig_h,
    panA.move(x=(w_AB - panA.width) / 2 + (fig_w - w_AB - panC.width - w_gap) / 2, y=0),
    panB.move(
        x=(w_AB - panB.width) / 2 + (fig_w - w_AB - panC.width - w_gap) / 2,
        y=h_upper - panB.height,
    ),
    panC.move(
        x=fig_w - panC.width - (fig_w - w_AB - panC.width - w_gap) / 2,
        y=(h_upper - panC.height) / 2,
    ),
    panD.move(x=(fig_w - panD.width) / 2, y=h_upper + h_gap),
)
fig.save(os.path.join(OUT_PATH, "fig2.svg"))
