import numpy as np
import pandas as pd


def compute_prop_weight(df: pd.DataFrame) -> pd.Series:
    dep_start = df[df["supplement"] != "water"]["date"].min()
    baseline = df[df["date"] < dep_start]["weight"].mean()
    df["prop_weight"] = df["weight"] / baseline
    return df