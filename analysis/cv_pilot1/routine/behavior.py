import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.ndimage import label


def merge_ts(ms_ts: pd.DataFrame, behav_ts: pd.DataFrame) -> pd.DataFrame:
    ms_ts["camNum"] = 0
    behav_ts["camNum"] = 1
    return pd.concat([ms_ts, behav_ts]).rename(
        columns={"Frame Number": "frameNum", "Time Stamp (ms)": "sysClock"}
    )


def map_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """
    map frames from Cam1 to Cam0 with nearest neighbour using the timestamp
    file from miniscope recordings.
    Parameters
    ----------
    ts : pd.DataFrame
        input timestamp dataframe. should contain field 'frameNum', 'camNum' and
        'sysClock'
    Returns
    -------
    pd.DataFrame
        output dataframe. should contain field 'fmCam0' and 'fmCam1'
    """
    ts_sort = ts.sort_values("sysClock")
    ts_sort["ts_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["sysClock"], np.nan)
    ts_sort["ts_forward"] = ts_sort["ts_behav"].fillna(method="ffill")
    ts_sort["ts_backward"] = ts_sort["ts_behav"].fillna(method="bfill")
    ts_sort["diff_forward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_forward"])
    ts_sort["diff_backward"] = np.absolute(ts_sort["sysClock"] - ts_sort["ts_backward"])
    ts_sort["fm_behav"] = np.where(ts_sort["camNum"] == 1, ts_sort["frameNum"], np.nan)
    ts_sort["fm_forward"] = ts_sort["fm_behav"].fillna(method="ffill")
    ts_sort["fm_backward"] = ts_sort["fm_behav"].fillna(method="bfill")
    ts_sort["fmCam1"] = np.where(
        ts_sort["diff_forward"] < ts_sort["diff_backward"],
        ts_sort["fm_forward"],
        ts_sort["fm_backward"],
    )
    ts_map = (
        ts_sort[ts_sort["camNum"] == 0][["frameNum", "fmCam1"]]
        .dropna()
        .rename(columns=dict(frameNum="fmCam0"))
        .astype(dict(fmCam1=int))
    )
    return ts_map


def linearize_pos(xy):
    a, b = np.polyfit(xy[:, 0], xy[:, 1], 1)
    xy[:, 1] = xy[:, 1] - b
    v = [1, a]
    v = v / np.linalg.norm(v)
    return (xy * v[np.newaxis, :]).sum(axis=1)


def code_direction(pos, smooth, diff_thres=None):
    pos_smooth = medfilt(pos, smooth)
    diff = np.diff(pos_smooth)
    diff = np.concatenate([[diff[0]], diff])
    pos_sign = pos_smooth * np.sign(diff)
    if diff_thres is not None:
        pos_sign = np.where(np.abs(diff) > diff_thres, pos_sign, np.nan)
    return pos_sign


def determine_trial(pos, smooth, rw_low=10, rw_high=90, min_time=2 * 60):
    pos = medfilt(pos, smooth)
    rw = np.full_like(pos, np.nan)
    rw = np.where(pos < rw_low, 0, rw)
    rw = np.where(pos > rw_high, 1, rw)
    start = rw[~np.isnan(rw)][0]
    trial, ntrial = label(rw == start)
    for t in range(ntrial):
        tidx = np.where(trial == t + 1)[0]
        trial[tidx] = 0
        if len(tidx) > min_time:
            trial[tidx[0]] = 1
    return trial.cumsum()
