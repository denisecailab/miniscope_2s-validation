import cv2
import numpy as np
import xarray as xr
from minian.cnmf import label_connected
from minian.initialization import local_max_roll
from scipy.optimize import minimize_scalar
from skimage.measure import label as imlabel
from sklearn.metrics import pairwise_distances

from .utilities import norm


def find_seed(max_proj: xr.DataArray, wnd_k0=8, wnd_k1=10, diff_thres=2):
    loc_max = xr.apply_ufunc(
        local_max_roll,
        max_proj,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.uint8],
        kwargs=dict(k0=wnd_k0, k1=wnd_k1 + 1, diff=diff_thres),
    )
    seeds = (
        loc_max.where(loc_max > 0).rename("seeds").to_dataframe().dropna().reset_index()
    )
    return seeds[["height", "width", "seeds"]]


def convexity_score(im):
    cnt = cv2.findContours(
        im.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[0][0]
    peri = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    peri_hull = cv2.arcLength(hull, True)
    if peri > 0:
        return peri_hull / peri
    else:
        return 0


def im_floodfill(im):
    im_floodfill = im.astype(np.uint8)
    h, w = im.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    return im_floodfill != 255


def cvx_opt_cb(thres, proj, min_size, pad, sd_h, sd_w):
    im_labs = proj < thres
    im_labs[max(sd_h - pad, 0) : sd_h + pad, max(sd_w - pad, 0) : sd_w + pad] = True
    im_labs = imlabel(im_labs)
    labs = im_labs[sd_h, sd_w]
    im_sd = im_labs == labs
    size = im_sd.sum()
    if size > min_size:
        return -convexity_score(im_sd)
    else:
        return 0


def constructA(seeds, max_proj, min_size=8 * 8, max_size=20 * 20, pad=2):
    img = np.array(max_proj)
    dx = cv2.Sobel(img, ddepth=-1, dx=1, dy=0)
    dy = cv2.Sobel(img, ddepth=-1, dx=0, dy=1)
    mag = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx)
    A_ls = []
    idx_ls = []
    for isd, row in seeds.iterrows():
        sd_h, sd_w = int(row["height"]), int(row["width"])
        gy = np.tile(np.arange(img.shape[0])[:, np.newaxis], (1, img.shape[1])) - sd_h
        gx = np.tile(np.arange(img.shape[1])[np.newaxis, :], (img.shape[0], 1)) - sd_w
        gang = np.arctan2(gy, gx)
        proj = mag * np.cos(gang - ang)
        res = minimize_scalar(
            cvx_opt_cb,
            bounds=(-200, 5),
            args=(proj, min_size, pad, sd_h, sd_w),
            method="Bounded",
        )
        if res.success:
            im_labs = proj < res.x
            im_labs[
                max(sd_h - pad, 0) : sd_h + pad, max(sd_w - pad, 0) : sd_w + pad
            ] = True
            im_labs = imlabel(im_labs)
            labs = im_labs[sd_h, sd_w]
            mask = im_floodfill(im_labs == labs)
            if mask.sum() <= max_size:
                curA = np.where(mask, max_proj, np.nan)
                curA[np.isnan(curA)] = np.nanmin(curA)
                A_ls.append(norm(curA))
                idx_ls.append(isd)
    A = xr.DataArray(
        np.stack(A_ls, axis=0),
        dims=["unit_id", "height", "width"],
        coords={
            "unit_id": idx_ls,
            "height": max_proj.coords["height"],
            "width": max_proj.coords["width"],
        },
    )
    return A


def mergeA(A, cos_thres=0.5):
    cos = 1 - pairwise_distances(
        np.array(A).reshape((A.shape[0], -1)), metric="cosine", n_jobs=-1
    )
    np.fill_diagonal(cos, 0)
    lab = label_connected(cos >= cos_thres)
    A_merged = (
        A.assign_coords(unit_labels=("unit_id", lab))
        .groupby("unit_labels")
        .mean("unit_id")
        .rename(unit_labels="unit_id")
    )
    return A_merged
