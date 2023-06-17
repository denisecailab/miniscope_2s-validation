import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from scipy.stats import gaussian_kde
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
    fr = df[fr_name].fillna(0)
    if fr.sum() > 0:
        if method == "com":
            return center_of_mass(fr)[0]
        elif method == "max":
            return df[space_name].iloc[fr.argmax()]
    else:
        return np.nan


def aggregate_fr(S_df, occp, bw, smp_space):
    fr = (
        S_df.groupby(["animal", "session", "trial", "unit_id"])
        .apply(
            kde_est,
            var_name="linpos",
            bandwidth=bw,
            smp_space=smp_space,
            weight_name="S",
        )
        .rename(columns={"linpos": "fr"})
        .reset_index()
    )
    fr_df = fr.merge(occp, how="left", on=["animal", "session", "trial", "smp_space"])[
        ["animal", "session", "trial", "unit_id", "smp_space", "fr", "occp"]
    ]
    fr_df = fr_df[fr_df["occp"].notnull()]
    fr_df["fr"] = fr_df["fr"].fillna(0)
    fr_df["fr_norm"] = np.nan_to_num(fr_df["fr"] / fr_df["occp"], posinf=0)
    return fr_df


def compute_metrics(fr_df):
    fr_agg = (
        fr_df.groupby(["animal", "session", "unit_id", "smp_space"])
        .agg({"fr_norm": "mean", "occp": "mean", "fr": "mean"})
        .reset_index()
    )
    stb_df = (
        fr_df.groupby(["animal", "session", "unit_id"])
        .apply(compute_stb)
        .rename("stb")
        .reset_index()
    )
    pos_df = (
        fr_agg.groupby(["animal", "session", "unit_id"])
        .apply(find_peak_field)
        .rename("peak")
        .reset_index()
    )
    si_df = (
        fr_agg.groupby(["animal", "session", "unit_id"])
        .apply(compute_si)
        .rename("si")
        .reset_index()
    )
    metric_df = stb_df.merge(
        pos_df, on=["animal", "session", "unit_id"], validate="one_to_one"
    ).merge(si_df, on=["animal", "session", "unit_id"], validate="one_to_one")
    return metric_df


def rollS(df):
    S = np.array(df["S"])
    df["S"] = np.roll(S, np.random.randint(len(S)))
    return df


def shuffleS(df, grp_by=["animal", "session", "unit_id", "trial"]):
    return df.groupby(
        ["animal", "session", "unit_id", "trial"], group_keys=False
    ).apply(rollS)


def classify_cell(df, stb_thres, si_thres):
    met_org = df[df["ishuf"] == -1].squeeze()
    met_shuf = df[df["ishuf"] >= 0]
    stb_org, si_org, peak_org = met_org["stb"], met_org["si"], met_org["peak"]
    stb_q = np.mean(stb_org > met_shuf["stb"].dropna())
    si_q = np.mean(si_org > met_shuf["si"].dropna())
    stb_sig, si_sig = stb_q >= stb_thres, si_q >= si_thres
    return pd.Series(
        {
            "stb": stb_org,
            "si": si_org,
            "peak": peak_org,
            "stb_q": stb_q,
            "si_q": si_q,
            "stb_sig": stb_sig,
            "si_sig": si_sig,
            "sig": stb_sig and si_sig,
        }
    )
