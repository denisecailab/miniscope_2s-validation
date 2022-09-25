#%% imports and definitions
import os
import pickle

import colorcet as cc
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

from routine.alignment import apply_affine, est_affine

hv.notebook_extension("bokeh")

IN_GREEN_PATH = "./intermediate/processed/green/m15-rec1.nc"
IN_RED_PATH = "./intermediate/processed/red/m15-rec1.nc"
OUT_TX = "./intermediate/alignment/tx.pkl"
OUT_DS = "./intermediate/alignment/ds.nc"
OUT_FIG = "./figs/alignment/m15.html"
FIG_PATH = "./figs/overlap"
CLIP = (5, 25)
BRT_OFFSET = 0
TITLES = {
    "top": "tdTomato",
    "side": "GCamp",
    "ovly": "Overlay",
    "top_max": "tdTomato Max Projection",
    "ovly_max": "Overlay",
}
os.makedirs(os.path.dirname(OUT_DS), exist_ok=True)
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


#%% compute alignment
fm_side = xr.open_dataset(IN_GREEN_PATH)["max_proj"]
fm_side = np.flip(fm_side, axis=0).assign_coords(
    height=np.arange(fm_side.sizes["height"])
)
fm_top = xr.open_dataset(IN_RED_PATH)["max_proj"]
tx, param_df = est_affine(fm_side.values, fm_top.values, lr=1)
fm_side_reg = xr.apply_ufunc(
    apply_affine,
    fm_side,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    kwargs={"tx": tx},
)
# plot result
im_opts = {"cmap": "viridis"}
hv_align = (
    hv.Image(fm_top, ["width", "height"], label="top").opts(**im_opts)
    + hv.Image(fm_side, ["width", "height"], label="side").opts(**im_opts)
    + hv.Image(fm_side_reg, ["width", "height"], label="side_reg").opts(**im_opts)
    + hv.Image((fm_side_reg - fm_top), ["width", "height"], label="affine").opts(
        **im_opts
    )
).cols(2)
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
hv.save(hv_align, OUT_FIG)
# save result
os.makedirs(os.path.dirname(OUT_TX), exist_ok=True)
ds = xr.merge(
    [
        fm_top.rename("fm_top"),
        fm_side.rename("fm_side"),
        fm_side_reg.rename("fm_side_reg"),
    ]
)
ds.to_netcdf(OUT_DS)
with open(OUT_TX, "wb") as pklf:
    pickle.dump(tx, pklf)

#%% plot overlap
fm_side = xr.open_dataset(IN_GREEN_PATH)["A"].max("unit_id")
fm_top = xr.open_dataset(IN_RED_PATH)["A"].max("unit_id")
fm_side = np.flip(fm_side, axis=0).assign_coords(
    height=np.arange(fm_side.sizes["height"])
)
fm_side = xr.apply_ufunc(
    apply_affine,
    fm_side,
    input_core_dims=[["height", "width"]],
    output_core_dims=[["height", "width"]],
    kwargs={"tx": tx},
)


def plot_im(a, ax):
    ax.imshow(a)
    ax.set_axis_off()


plt.rcParams.update({"axes.titlesize": 11, "font.sans-serif": "Arial"})
aspect = 1.4
fig, axs = plt.subplots(1, 3, figsize=(8.5, 8.5 / aspect), dpi=500)
ax_top, ax_side, ax_ovly = (axs[0], axs[1], axs[2])
fm_top_pcolor = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(fm_top.values)
    + BRT_OFFSET,
    0,
    1,
)
fm_side_pcolor = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(fm_side.values)
    + BRT_OFFSET,
    0,
    1,
)
fm_ovly = np.clip(fm_top_pcolor + fm_side_pcolor, 0, 1)
plt.subplots_adjust(0, 0, 1, 1, 0.05, 0.05)
ax_top.set_title(TITLES["top"])
ax_side.set_title(TITLES["side"])
ax_ovly.set_title(TITLES["ovly"])
plot_im(fm_top_pcolor, ax_top)
plot_im(fm_side_pcolor, ax_side)
plot_im(fm_ovly, ax_ovly)
aspect = fm_top_pcolor.shape[1] / fm_top_pcolor.shape[0]
opts_im = {
    "frame_width": 380,
    "aspect": aspect,
    "xaxis": None,
    "yaxis": None,
    "fontsize": {"title": 15},
}
hv_plt = (
    hv.RGB(fm_top_pcolor, ["width", "height"], label=TITLES["top"]).opts(**opts_im)
    + hv.RGB(fm_side_pcolor, ["width", "height"], label=TITLES["side"]).opts(**opts_im)
    + hv.RGB(fm_ovly, ["width", "height"], label=TITLES["ovly"]).opts(**opts_im)
).cols(3)
hv.save(hv_plt, os.path.join(FIG_PATH, "m15.html"))
fig.savefig(os.path.join(FIG_PATH, "m15.png"), bbox_inches="tight", dpi=500)
