import re

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.signal import medfilt


def linearize_pos(xy):
    a, b = np.polyfit(xy[:, 0], xy[:, 1], 1)
    xy[:, 1] = xy[:, 1] - b
    v = [1, a]
    v = v / np.linalg.norm(v)
    return (xy * v[np.newaxis, :]).sum(axis=1)


def code_direction(pos, smooth, diff_thres=None):
    pos_smooth = medfilt(pos, smooth)
    diff = medfilt(np.gradient(pos_smooth, 5), smooth)
    pos_sign = pos_smooth * np.sign(diff)
    if diff_thres is not None:
        running_mask = np.zeros_like(pos_sign, dtype=bool)
        diff_mag = np.abs(diff)
        for rng, thres in diff_thres.items():
            running_mask[
                np.logical_and.reduce(
                    [pos_smooth >= rng[0], pos_smooth <= rng[1], diff_mag > thres]
                )
            ] = True
        pos_sign = np.where(running_mask, pos_sign, np.nan)
    return pos_sign


def determine_trial(df):
    rw_first = df[df["event"] == "REWARD"].iloc[0]
    df["trial"] = 0
    df.loc[(df["event"] == "REWARD") & (df["data"] == rw_first["data"]), "trial"] = 1
    df["trial"] = df["trial"].cumsum()
    return df


def extract_location(r):
    if type(r["data"]) is not str:
        return pd.Series([], dtype=float)
    match = re.search(r"X(?P<x>\d+)Y(?P<y>\d+)", r["data"])
    if match:
        return pd.Series(match.groupdict())
    else:
        return pd.Series([], dtype=float)


def label_ts(behav, ms_ts):
    behav = behav.copy()
    ms_ts = ms_ts.copy()
    tstart = behav[behav["event"] == "START"]["timestamp"].item()
    tterminate = behav[behav["event"] == "TERMINATE"]["timestamp"].item()
    behav = behav[behav["timestamp"].between(tstart, tterminate, inclusive="both")]
    behav["timestamp"] = (behav["timestamp"] - tterminate) * 1e3
    ms_ts["timestamp"] = ms_ts["timestamp"] - ms_ts["timestamp"].iloc[-1].item()
    fm_cut = (ms_ts["timestamp"].values[1:] + ms_ts["timestamp"].values[:-1]) / 2
    fm_cut = np.append(np.insert(fm_cut, 0, ms_ts["timestamp"].iloc[0].item()), 0)
    behav["ms_frame"] = pd.cut(behav["timestamp"], fm_cut, labels=ms_ts["frame"])
    return behav[behav["ms_frame"].notnull()]


def agg_behav(df):
    if len(df) > 0:
        return pd.Series(
            {
                "x": df["x"].median(),
                "y": df["y"].median(),
                "trial": df["trial"].mode().values[0],
            }
        )
    else:
        return pd.Series(
            data=np.full(3, np.nan),
            index=["x", "y", "trial"],
        )
