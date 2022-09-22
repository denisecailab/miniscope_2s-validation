import itertools as itt
import os
import re

import numpy as np
from scipy.interpolate import interp1d


def norm(a):
    amax, amin = np.nanmax(a), np.nanmin(a)
    return (a - amin) / (amax - amin)


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
