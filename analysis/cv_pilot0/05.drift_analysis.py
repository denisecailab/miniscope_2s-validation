#%% imports and definitions
import itertools as itt
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from plotly.express.colors import qualitative
from tqdm.auto import tqdm

from routine.place_cell import kde_est
from routine.plotting import scatter_agg
from routine.utilities import nan_corr

IN_SS_CSV = "./log/sessions.csv"
IN_PS_PATH = "./intermediate/processed/green"
IN_FM_LABEL = "./intermediate/frame_label/fm_label.nc"
IN_RAW_MAP = "./intermediate/cross_reg/green/mappings_meta.pkl"
IN_REG_MAP = "./intermediate/register_g2r/green_mapping_reg.pkl"
PARAM_BW = 3
PARAM_SMP_SPACE = np.linspace(-100, 100, 200)
OUT_PATH = "./intermediate/drift"
FIG_PATH = "./figs/drift"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% compute fr
ss_csv = pd.read_csv(IN_SS_CSV)
ss_csv = ss_csv[ss_csv["analyze"]]
fm_label = xr.open_dataset(IN_FM_LABEL)
fr_ls = []
for _, row in tqdm(list(ss_csv.iterrows())):
    anm, ss = row["animal"], row["name"]
    try:
        ps_ds = xr.open_dataset(
            os.path.join(IN_PS_PATH, "{}-{}.nc".format(anm, ss))
        ).assign_coords(linpos=fm_label["linpos_sign"])
    except FileNotFoundError:
        warnings.warn("missing data for {}-{}".format(anm, ss))
        continue
    S_df = ps_ds["S"].to_dataframe().dropna().reset_index()
    pos_df = S_df[["animal", "session", "frame", "linpos"]].drop_duplicates()
    occp = (
        pos_df.groupby(["animal", "session"])
        .apply(
            kde_est,
            var_name="linpos",
            bw_method=PARAM_BW,
            smp_space=PARAM_SMP_SPACE,
            zero_thres=1e-4,
        )
        .rename(columns={"linpos": "occp"})
        .reset_index()
    )
    fr = (
        S_df.groupby(["animal", "session", "unit_id"])
        .apply(
            kde_est,
            var_name="linpos",
            bw_method=PARAM_BW,
            smp_space=PARAM_SMP_SPACE,
            weight_name="S",
        )
        .rename(columns={"linpos": "fr"})
        .reset_index()
    )
    fr_df = fr.merge(occp, how="left", on=["animal", "session", "smp_space"])[
        ["animal", "session", "unit_id", "smp_space", "fr", "occp"]
    ]
    fr_df["fr_norm"] = fr_df["fr"] / fr_df["occp"]
    fr_ls.append(fr_df)
fr_df = pd.concat(fr_ls, ignore_index=True)
fr_df.to_feather(os.path.join(OUT_PATH, "fr.feat"))

#%% compute pv corr
ss_csv = (
    pd.read_csv(IN_SS_CSV, parse_dates=["date"])
    .sort_values(["animal", "name"])
    .set_index(["animal", "name"])
)
mapping_dict = {
    "green/raw": pd.read_pickle(IN_RAW_MAP),
    "red/registered": pd.read_pickle(IN_REG_MAP),
}
fr_df = pd.read_feather(os.path.join(OUT_PATH, "fr.feat")).set_index(
    ["animal", "session", "unit_id"]
)
pv_corr_ls = []
for mmethod, mmap in mapping_dict.items():
    for ssA, ssB in tqdm(
        list(itt.combinations(mmap["session"].columns, 2)), leave=False
    ):
        mmap_sub = mmap[[("meta", "animal"), ("session", ssA), ("session", ssB)]]
        mmap_sub = mmap_sub[(mmap_sub["session"] >= 0).all(axis="columns")]
        mmap_sub.columns = mmap_sub.columns.droplevel(0)
        for anm, mp in mmap_sub.groupby("animal"):
            tdist = np.abs(
                (
                    ss_csv.loc[anm, ssA]["date"].iloc[0]
                    - ss_csv.loc[anm, ssB]["date"].iloc[0]
                ).days
            )
            frA = (
                fr_df.loc[anm, ssA, mp[ssA].values]
                .reset_index()
                .set_index(["unit_id", "smp_space"])["fr_norm"]
                .to_xarray()
            )
            frB = (
                fr_df.loc[anm, ssB, mp[ssB].values]
                .reset_index()
                .set_index(["unit_id", "smp_space"])["fr_norm"]
                .to_xarray()
            )
            r = nan_corr(frA, frB)
            pv_corr_ls.append(
                pd.DataFrame(
                    [
                        {
                            "map_method": mmethod,
                            "animal": anm,
                            "ssA": ssA,
                            "ssB": ssB,
                            "tdist": tdist,
                            "corr": r,
                        }
                    ]
                )
            )
pv_corr = pd.concat(pv_corr_ls, ignore_index=True)
pv_corr.to_csv(os.path.join(OUT_PATH, "pv_corr.csv"))

#%% plot result
cmap = {"green/raw": qualitative.Plotly[0], "red/registered": qualitative.Plotly[1]}
pv_corr = pd.read_csv(os.path.join(OUT_PATH, "pv_corr.csv"))
pv_corr["color"] = pv_corr["map_method"].map(cmap)
fig = scatter_agg(
    pv_corr,
    x="tdist",
    y="corr",
    facet_row=None,
    facet_col="animal",
    col_wrap=3,
    legend_dim="map_method",
    marker={"color": "color"},
)
fig.update_xaxes(title="Days apart")
fig.update_yaxes(range=(-0.1, 0.6), title="PV correlation")
fig.write_html(os.path.join(FIG_PATH, "pv_corr.html"))
fig = scatter_agg(
    pv_corr,
    x="tdist",
    y="corr",
    facet_row=None,
    facet_col=None,
    legend_dim="map_method",
    marker={"color": "color"},
)
fig.update_xaxes(title="Days apart")
fig.update_yaxes(range=(-0.1, 0.6), title="PV correlation")
fig.write_html(os.path.join(FIG_PATH, "pv_corr_master.html"))
