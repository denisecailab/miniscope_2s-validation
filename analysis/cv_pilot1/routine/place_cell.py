import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import center_of_mass
from sklearn.neighbors import KernelDensity

from .utilities import nan_corr


def kde_est(
    data: pd.DataFrame,
    var_name: str,
    bandwidth: float,
    smp_space: np.ndarray,
    kernel="cosine",
    weight_name: str = None,
    spk_count_thres: int = 0,
    **kwargs,
) -> pd.DataFrame:
    dat = data[var_name]
    if dat.nunique() > 1:
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, **kwargs)
        dat = np.array(dat)
        if weight_name is not None:
            wt = np.array(data[weight_name])
            if (wt > 0).sum() > spk_count_thres:
                dat = dat[wt > 0]
                wt = wt[wt > 0]
                kde.fit(dat.reshape((-1, 1)), sample_weight=wt)
            else:
                return pd.DataFrame({"smp_space": smp_space, var_name: np.nan})
        else:
            kde.fit(dat.reshape((-1, 1)))
        density = np.exp(kde.score_samples(smp_space.reshape((-1, 1))))
        if density.sum() == 0:
            density = np.nan
    else:
        density = np.nan
    return pd.DataFrame({"smp_space": smp_space, var_name: density})


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
    if first.max() > 0 and last.max() > 0:
        r_fl = nan_corr(first, last)
    else:
        r_fl = np.nan
    if odd.max() > 0 and even.max() > 0:
        r_oe = nan_corr(odd, even)
    else:
        r_oe = np.nan
    rs = np.array([r_fl, r_oe])
    if np.isnan(rs).all():
        return np.nan
    else:
        return np.nanmean(rs)


def compute_si(df, fr_name="fr", occp_name="occp") -> float:
    fr, occp = df[fr_name].values, df[occp_name].values
    mfr = fr.mean()
    if mfr > 0:
        fr_norm = fr / mfr
        return (occp * fr_norm * np.log2(fr_norm, where=fr_norm > 0)).sum()
    else:
        return np.nan


def find_peak_field(df, fr_name="fr_norm", space_name="smp_space", method="com"):
    if method == "com":
        return center_of_mass(df[fr_name])[0]
    elif method == "max":
        return df[space_name].iloc[df[fr_name].argmax()]
