import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import center_of_mass

from .utilities import corr_mat


def kde_est(
    data: pd.DataFrame,
    var_name: str,
    bw_method: float,
    smp_space: np.ndarray,
    weight_name: str = None,
    zero_thres: float = 0,
    spk_count_thres: int = 5,
) -> pd.DataFrame:
    if data[var_name].nunique() > 1:
        if weight_name is not None:
            if (data[weight_name] > 0).sum() > spk_count_thres:
                kernel = gaussian_kde(
                    data[var_name], bw_method=bw_method, weights=data[weight_name]
                )
            else:
                return pd.DataFrame({"smp_space": smp_space, var_name: np.nan})
        else:
            kernel = gaussian_kde(data[var_name], bw_method=bw_method)
        kde = kernel(smp_space)
        kde[kde < zero_thres] = 0
        if kde.sum() == 0:
            kde = np.nan
    else:
        kde = np.nan
    return pd.DataFrame({"smp_space": smp_space, var_name: kde})


def compute_stb(
    df, fr_name="fr_norm", trial_name="trial", space_name="smp_space"
) -> float:
    trials = np.sort(df[trial_name].unique())
    ntrial = len(trials)
    df["trial_idx"] = df[trial_name].map(
        {k: v for k, v in zip(trials, np.arange(ntrial))}
    )
    ntrial = ntrial - 1 if ntrial % 2 == 1 else ntrial
    df = df[df["trial_idx"] < ntrial].copy()
    if not len(df) > 0:
        return np.nan
    first = (
        df[df["trial_idx"] < ntrial / 2]
        .set_index([space_name, trial_name])[fr_name]
        .to_xarray()
        .values
    )
    last = (
        df[df["trial_idx"] >= ntrial / 2]
        .set_index([space_name, trial_name])[fr_name]
        .to_xarray()
        .values
    )
    odd = (
        df[df["trial_idx"] % 2 == 1]
        .set_index([space_name, trial_name])[fr_name]
        .to_xarray()
        .values
    )
    even = (
        df[df["trial_idx"] % 2 == 0]
        .set_index([space_name, trial_name])[fr_name]
        .to_xarray()
        .values
    )
    rs = np.array([corr_mat(first, last), corr_mat(odd, even)])
    if np.isnan(rs).all():
        return np.nan
    else:
        return np.nanmean(rs)


def compute_si(df, fr_name="fr_norm", occp_name="occp") -> float:
    fr, occp = df[fr_name].values, df[occp_name].values
    mfr = fr.mean()
    fr_norm = fr / mfr
    return (occp * fr_norm * np.log2(fr_norm, where=fr_norm > 0)).sum()


def find_peak_field(df, fr_name="fr_norm", space_name="smp_space", method="com"):
    if method == "com":
        return center_of_mass(df[fr_name])[0]
    elif method == "max":
        return df[space_name].iloc[df[fr_name].argmax()]
