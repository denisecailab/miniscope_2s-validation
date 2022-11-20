#%% imports and definition
import os

from svgutils.compose import SVG, Figure, Panel, Text

PARAMT_TEXT = {"size": 11, "weight": "bold"}
OUT_PATH = "./figs/20221110_fluidic_grant"
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
w_gap = 10
h_gap = 1
panA = make_panel("A", "./figs/external/linear_track.svg", 0.5)
panB = make_panel("B", "./figs/frame_label/example.svg")
panC = make_panel("C", "./figs/register_g2r/summary.svg")
panD = make_panel("D", "./figs/register_g2r/cells/m15.svg")
w_left = max(panA.width, panC.width)
w_right = max(panB.width, panD.width)
h_up = max(panA.height, panB.height)
h_bot = max(panC.height, panD.height)
fig_h = h_up + h_gap + h_bot
fig_w = w_left + w_gap + w_right
# fig = Figure(
#     fig_w,
#     fig_h,
#     panA.move(x=(w_left - panA.width) / 2, y=(h_up - panA.height) / 2),
#     panB.move(
#         x=(w_right - panB.width) / 2 + w_left + w_gap, y=(h_up - panB.height) / 2
#     ),
#     panC.move(x=(w_left - panC.width) / 2, y=(h_bot - panC.height) / 2 + h_up + h_gap),
#     panD.move(
#         x=(w_right - panD.width) / 2 + w_left + w_gap,
#         y=(h_bot - panD.height) / 2 + h_up + h_gap,
#     ),
# )
fig = Figure(
    fig_w,
    fig_h,
    panA,
    panB.move(x=w_left + w_gap, y=0),
    panC.move(x=0, y=h_up + h_gap),
    panD.move(x=w_left + w_gap, y=h_up + h_gap),
)
fig.save(os.path.join(OUT_PATH, "grant_figure.svg"))
