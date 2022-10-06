import colorcet as cc
import cv2
import holoviews as hv
import numpy as np
import xarray as xr
from matplotlib import cm

hv.notebook_extension("bokeh")


def plotA_contour(A: xr.DataArray, im: xr.DataArray):
    im = hv.Image(im, ["width", "height"])
    for uid in A.coords["unit_id"].values:
        curA = (np.array(A.sel(unit_id=uid)) > 0).astype(np.uint8)
        cnt = cv2.findContours(curA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][
            0
        ].squeeze()
        if cnt.ndim > 1:
            im = im * hv.Path(cnt.squeeze())
    return im


def plot_overlap(
    im_green: np.ndarray, im_red: np.ndarray, brt_offset=0, return_raw=False
):
    im_red = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(
            np.array(im_red)
        )
        + brt_offset,
        0,
        1,
    )
    im_green = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(
            np.array(im_green)
        )
        + brt_offset,
        0,
        1,
    )
    ovly = np.clip(im_red + im_green, 0, 1)
    if return_raw:
        return ovly, im_green, im_red
    else:
        return ovly
