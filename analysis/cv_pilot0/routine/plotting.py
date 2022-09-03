import cv2
import holoviews as hv
import numpy as np
import xarray as xr

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
