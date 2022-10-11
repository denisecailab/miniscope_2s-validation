import itertools as itt
import os
import re

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


def norm(a):
    amax, amin = np.nanmax(a), np.nanmin(a)
    return (a - amin) / (amax - amin)


def norm_xr(x: xr.DataArray, q=1):
    xmin = x.min().compute().values
    if q < 1:
        xmax = x.compute().quantile(q).compute().values
    else:
        xmax = x.max().compute().values
    diff = xmax - xmin
    if diff > 0:
        return ((x - xmin) / diff).clip(0, 1)
    else:
        return x


def walklevel(path, depth=1):
    if depth < 0:
        for root, dirs, files in os.walk(path):
            yield root, dirs[:], files
        return
    elif depth == 0:
        return
    base_depth = path.rstrip(os.path.sep).count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs[:], files
        cur_depth = root.count(os.path.sep)
        if base_depth + depth <= cur_depth:
            del dirs[:]


def subset_sessions(r, anm_pat, ss_pat):
    anm_match = re.search(anm_pat, r["animal"])
    ss_match = re.search(ss_pat, r["name"])
    return bool(anm_match) and bool(ss_match)


def xr_reset_index(ds, dim):
    if type(dim) is not list:
        dim = [dim]
    for d in dim:
        ds = ds.assign_coords({d: np.arange(ds.sizes[d])})
    return ds


def xr_nditer(arr, dims):
    for key in itt.product(*[arr.coords[d].values for d in dims]):
        asub = arr.sel({d: k for d, k in zip(dims, key)}).squeeze()
        for d in asub.dims:
            asub = asub.dropna(d, how="all")
        if len(asub) > 0:
            yield (key, asub)


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def resample_motion(motion, nsmp):
    motion_ret = np.zeros((nsmp, 2))
    f_org = np.linspace(0, 1, motion.shape[0], endpoint=True)
    f_new = np.linspace(0, 1, nsmp, endpoint=True)
    for i in range(2):
        motion_ret[:, i] = interp1d(f_org, motion[:, i])(f_new)
    return motion_ret


def df_set_metadata(dfs: list, meta: dict):
    if type(dfs) is not list:
        return_list = False
        dfs = [dfs]
    else:
        return_list = True
    for df in dfs:
        for k, v in meta.items():
            df[k] = v
    if return_list:
        return dfs
    else:
        return dfs[0]


def df_map_values(dfs: list, mappings: dict):
    if type(dfs) is not list:
        return_list = False
        dfs = [dfs]
    else:
        return_list = True
    for df in dfs:
        for col, mp in mappings.items():
            df[col] = df[mp[0]].map(mp[1])
    if return_list:
        return dfs
    else:
        return dfs[0]


def nan_corr(a, b):
    a, b = np.array(a).flatten(), np.array(b).flatten()
    mask = np.logical_and(~np.isnan(a), ~np.isnan(b))
    a, b = a[mask], b[mask]
    a = a - np.mean(a)
    b = b - np.mean(b)
    return (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum())
