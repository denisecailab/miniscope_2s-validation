import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


def calculate_mapping_linear_sum(
    centA, centB, feat_cols=["height", "width"], id_col="unit_id", thres=None
):
    dist = pairwise_distances(centA[feat_cols], centB[feat_cols], n_jobs=-1)
    if thres is not None:
        dist = np.where(dist < thres, dist, dist.max())
    idxA, idxB = linear_sum_assignment(dist)
    map_df = pd.DataFrame(
        {
            "idxA": np.array(centA[id_col])[idxA],
            "idxB": np.array(centB[id_col])[idxB],
            "distance": dist[idxA, idxB],
        }
    )
    if thres is not None:
        map_df = map_df[map_df["distance"] < thres].copy()
    return map_df
