import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def kde_est(
    data: pd.DataFrame,
    var_name: str,
    bw_method: float,
    smp_space: np.ndarray,
    weight_name: str = None,
    zero_thres: float = 0,
) -> pd.DataFrame:
    if data[var_name].nunique() > 1:
        if weight_name is not None:
            if (data[weight_name] > 0).sum() > 1:
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
