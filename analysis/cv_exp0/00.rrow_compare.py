# %% imports and definitions
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plotly.express.colors import qualitative
from scipy.stats import ttest_rel

IN_DPATH = "./data-rrow/"
FIG_PATH = "./figs/behav_comparison/"
PARAM_PLT_RC = {
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "font.sans-serif": "Arial",
}
PARAM_CMAP = {
    "Dual-channel\nMiniscope": qualitative.Plotly[5],
    "Single-channel\nMiniscope": qualitative.Plotly[8],
}
plt.rcParams.update(**PARAM_PLT_RC)


def cap_metrics(met: str):
    if met == "distance traveled (m)":
        return "Distance Travelled (m)"
    else:
        return met.title()


# %% plot data
df_within = (
    pd.read_csv(os.path.join(IN_DPATH, "within.csv"))
    .rename(columns={"miniscope": "group", "mouse ID": "animal"})
    .melt(id_vars=["animal", "group"], var_name="metric", value_name="value")
)
df_plt = df_within.copy()
df_plt["metric"] = df_plt["metric"].map(cap_metrics)
df_plt["group"] = df_plt["group"].replace(
    {"2c ": "Dual-channel\nMiniscope", "1c ": "Single-channel\nMiniscope"}
)
g = sns.FacetGrid(
    df_plt,
    col="metric",
    sharey=False,
    height=2.5,
    aspect=0.95,
)
g.map_dataframe(
    sns.barplot,
    x="group",
    y="value",
    hue="group",
    palette=PARAM_CMAP,
    errorbar="se",
    err_kws={"linewidth": 3},
    capsize=0.2,
    saturation=0.9,
    alpha=0.75,
    width=0.5,
)
g.map_dataframe(
    sns.swarmplot,
    x="group",
    y="value",
    hue="group",
    palette=PARAM_CMAP,
    linewidth=1.2,
    warn_thresh=0.8,
    edgecolor="auto",
    alpha=0.9,
)
g.map_dataframe(
    sns.lineplot,
    x="group",
    y="value",
    linewidth=1.5,
    linestyle="dotted",
    units="animal",
    color="black",
    alpha=0.4,
    estimator=None,
    hue=None,
    zorder=1,
)
g.set_titles(col_template="{col_name}")
g.set_axis_labels(x_var="", y_var="")
g.despine(top=False, right=False)
g.figure.savefig(os.path.join(FIG_PATH, "rrow.svg"), dpi=500, bbox_inches="tight")

# %% t tests
for met, met_df in df_within.groupby("metric"):
    met_df = met_df.sort_values("animal")
    res = ttest_rel(
        met_df.loc[met_df["group"] == "1c ", "value"],
        met_df.loc[met_df["group"] == "2c ", "value"],
    )
    print(f"{met} pval: {res.pvalue:.3f}")
