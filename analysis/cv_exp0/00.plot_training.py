# %% import and definitions
import os

import numpy as np
import pandas as pd
import plotly.express as px
from routine.training import compute_prop_weight
from scipy.stats import zscore

IN_SS_PATH = "./log/sessions.csv"
IN_WT_PATH = "./log/weight.csv"
FIG_PATH = "./figs/training"

os.makedirs(FIG_PATH, exist_ok=True)

# %% plot weight
wt_df = pd.read_csv(IN_WT_PATH, parse_dates=["date"])
wt_df = wt_df.groupby("animal", group_keys=False).apply(compute_prop_weight)
fig = px.line(wt_df, x="date", y="weight", color="animal")
fig.write_html(os.path.join(FIG_PATH, "weight.html"))
fig = px.line(wt_df, x="date", y="prop_weight", color="animal")
fig.add_hline(0.85, line_dash="dash", line_color="grey")
fig.write_html(os.path.join(FIG_PATH, "prop_weight.html"))

# %% plot performance
ss_df = pd.read_csv(IN_SS_PATH, parse_dates=["date"])
ss_df = ss_df[ss_df["reward"].notnull()].copy()
ss_df["reward_z"] = ss_df.groupby("animal")["reward"].transform(zscore)
fig = px.line(ss_df, x="date", y="reward", color="animal")
fig.write_html(os.path.join(FIG_PATH, "reward.html"))
fig = px.line(ss_df, x="date", y="reward_z", color="animal")
fig.write_html(os.path.join(FIG_PATH, "reward_z.html"))
