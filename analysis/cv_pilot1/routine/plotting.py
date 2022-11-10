import colorcet as cc
import cv2
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import xarray as xr
from matplotlib import cm
from plotly import graph_objects as go
from plotly.express.colors import qualitative

from .utilities import unique_seg

hv.notebook_extension("bokeh")


def plotA_contour(A: xr.DataArray, im: xr.DataArray, cmap=None, im_opts=None):
    im = hv.Image(im, ["width", "height"])
    if im_opts is not None:
        im = im.opts(**im_opts)
    im = im * hv.Path([])
    for uid in A.coords["unit_id"].values:
        curA = (np.array(A.sel(unit_id=uid)) > 0).astype(np.uint8)
        try:
            cnt = cv2.findContours(curA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][
                0
            ].squeeze()
        except IndexError:
            continue
        if cnt.ndim > 1:
            pth = hv.Path(cnt.squeeze())
            if cmap is not None:
                pth = pth.opts(color=cmap[uid])
            im = im * pth
    return im


def plot_overlap(
    im_green: np.ndarray, im_red: np.ndarray, brt_offset=0, return_raw=False
):
    im_red = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(
            np.array(im_red)
        )
        + brt_offset,
        0,
        1,
    )
    im_green = np.clip(
        cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42).to_rgba(
            np.array(im_green)
        )
        + brt_offset,
        0,
        1,
    )
    ovly = np.clip(im_red + im_green, 0, 1)
    if return_raw:
        return ovly, im_green, im_red
    else:
        return ovly


import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utilities import enumerated_product


def facet_plotly(
    data: pd.DataFrame,
    facet_row: str,
    facet_col: str,
    title_dim: str = None,
    specs: dict = None,
    col_wrap: int = None,
    **kwargs,
):
    row_crd = data[facet_row].unique()
    col_crd = data[facet_col].unique()
    layout_ls = []
    iiter = 0
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        dat_sub = data[(data[facet_row] == r) & (data[facet_col] == c)]
        if not len(dat_sub) > 0:
            continue
        if title_dim is not None:
            title = dat_sub[title_dim].unique().item()
        else:
            if facet_row == "DUMMY_FACET_ROW":
                title = "{}={}".format(facet_col, c)
            elif facet_col == "DUMMY_FACET_COL":
                title = "{}={}".format(facet_row, r)
            else:
                title = "{}={}; {}={}".format(facet_row, r, facet_col, c)
        if col_wrap is not None:
            ir = iiter // col_wrap
            ic = iiter % col_wrap
            iiter += 1
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": title}
        )
    layout = pd.DataFrame(layout_ls).set_index(["row_label", "col_label"])
    if col_wrap is not None:
        nrow, ncol = int(layout["row"].max() + 1), int(layout["col"].max() + 1)
    else:
        nrow, ncol = len(row_crd), len(col_crd)
    if specs is not None:
        specs = np.full((nrow, ncol), specs).tolist()
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        subplot_titles=layout["title"].values,
        specs=specs,
        **kwargs,
    )
    return fig, layout


def handle_single_facet(data, facet_row, facet_col):
    data = data.copy()
    if facet_row is None:
        data["DUMMY_FACET_ROW"] = ""
        facet_row = "DUMMY_FACET_ROW"
    if facet_col is None:
        data["DUMMY_FACET_COL"] = ""
        facet_col = "DUMMY_FACET_COL"
    return data, facet_row, facet_col


def scatter_3d(
    data, facet_row, facet_col, legend_dim: str = None, title_dim: str = None, **kwargs
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, specs={"type": "scene"}
    )
    if legend_dim is not None:
        show_legend = {l: True for l in data[legend_dim].unique()}
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        if legend_dim is not None:
            for llab, subdf in facet_df.groupby(legend_dim, observed=True):
                show_leg = show_legend[llab]
                cur_args = transform_arguments(subdf, kwargs)
                trace = go.Scatter3d(
                    name=llab, legendgroup=llab, showlegend=show_leg, **cur_args
                )
                if show_leg:
                    show_legend[llab] = False
                fig.add_trace(trace, row=ir + 1, col=ic + 1)
        else:
            cur_args = transform_arguments(facet_df, kwargs)
            trace = go.Scatter3d(showlegend=False, **cur_args)
            fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def transform_arguments(data: pd.DataFrame, arguments: dict):
    arg_ret = copy.deepcopy(arguments)
    if arg_ret.get("x"):
        arg_ret["x"] = data[arg_ret["x"]].values
    if arg_ret.get("y"):
        arg_ret["y"] = data[arg_ret["y"]].values
    if arg_ret.get("z"):
        arg_ret["z"] = data[arg_ret["z"]].values
    if arg_ret.get("error_x"):
        x_err = data[arg_ret["error_x"]].values
        if np.nansum(x_err) > 0:
            arg_ret["error_x"] = {"array": x_err}
        else:
            del arg_ret["error_x"]
    if arg_ret.get("error_y"):
        y_err = data[arg_ret["error_y"]].values
        if np.nansum(y_err) > 0:
            arg_ret["error_y"] = {"array": y_err}
        else:
            del arg_ret["error_y"]
    if arg_ret.get("text"):
        try:
            arg_ret["text"] = data[arg_ret["text"]].values
        except KeyError:
            del arg_ret["text"]
    mkopts = arg_ret.get("marker")
    if mkopts:
        if mkopts.get("color"):
            try:
                color = data[mkopts["color"]].values
                if len(np.unique(color)) == 1:
                    mkopts["color"] = np.unique(color).item()
                else:
                    mkopts["color"] = color
            except KeyError:
                pass
        if mkopts.get("size"):
            try:
                size = data[mkopts["size"]].values
                if len(np.unique(size)) == 1:
                    mkopts["size"] = np.unique(size).item()
                else:
                    mkopts["size"] = size
            except KeyError:
                pass
        if mkopts.get("symbol"):
            try:
                symb = data[mkopts["symbol"]].values
                if len(np.unique(symb)) == 1:
                    mkopts["symbol"] = np.unique(symb).item()
                else:
                    mkopts["symbol"] = symb
            except KeyError:
                pass
    return arg_ret


def scatter_agg(
    data,
    x,
    y,
    facet_row,
    facet_col,
    col_wrap: int = None,
    legend_dim: str = None,
    title_dim: str = None,
    show_point_legend=False,
    subplot_args: dict = dict(),
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(
        data, facet_row, facet_col, title_dim, col_wrap=col_wrap, **subplot_args
    )
    grp_dims = [facet_row, facet_col, x]
    idx_dims = [facet_row, facet_col]
    if legend_dim is not None:
        grp_dims.append(legend_dim)
        idx_dims.append(legend_dim)
        show_legend = {l: True for l in data[legend_dim].unique()}
    data_agg = (
        data.groupby(grp_dims)[y]
        .agg(["mean", "sem"])
        .reset_index()
        .merge(data, on=grp_dims)
        .set_index(idx_dims)
        .sort_index()
    )
    kwargs["x"] = x
    kwargs["y"] = y
    kwargs_agg = copy.deepcopy(kwargs)
    kwargs_agg["y"] = "mean"
    kwargs_agg["error_y"] = "sem"
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        trace_ls = []
        if legend_dim is not None:
            for llab, subdf in facet_df.groupby(legend_dim, observed=True):
                show_leg = show_legend[llab]
                cur_args = transform_arguments(subdf, kwargs)
                ndata = subdf.groupby(x)[y].count().max()
                if ndata > 1:
                    strp = go.Box(
                        name=llab,
                        legendgroup=llab,
                        showlegend=show_leg and show_point_legend,
                        boxpoints="all",
                        fillcolor="rgba(255,255,255,0)",
                        hoveron="points",
                        line={"color": "rgba(255,255,255,0)"},
                        pointpos=0,
                        opacity=0.4,
                        **cur_args,
                    )
                    trace_ls.append(strp)
                subdf_agg = data_agg.loc[rlab, clab, llab].reset_index()
                cur_args = transform_arguments(subdf_agg, kwargs_agg)
                ln = go.Scatter(
                    name=llab,
                    legendgroup=llab,
                    showlegend=show_leg,
                    mode="lines",
                    **cur_args,
                )
                trace_ls.append(ln)
                if show_leg:
                    show_legend[llab] = False
        else:
            cur_args = transform_arguments(facet_df, kwargs)
            ndata = facet_df.groupby(x)[y].count().max()
            if ndata > 1:
                strp = go.Box(
                    boxpoints="all",
                    fillcolor="rgba(255,255,255,0)",
                    hoveron="points",
                    line={"color": "rgba(255,255,255,0)"},
                    pointpos=0,
                    opacity=0.4,
                    **cur_args,
                )
                trace_ls.append(strp)
            facet_df_agg = data_agg.loc[rlab, clab].reset_index()
            cur_args = transform_arguments(facet_df_agg, kwargs_agg)
            ln = go.Scatter(mode="lines", **cur_args)
            trace_ls.append(ln)
        for t in trace_ls:
            fig.add_trace(t, row=ir + 1, col=ic + 1)
    return fig


def imshow(
    data,
    facet_row,
    facet_col,
    title_dim: str = None,
    subplot_args: dict = dict(),
    **kwargs,
):
    data, facet_row, facet_col = handle_single_facet(data, facet_row, facet_col)
    fig, layout = facet_plotly(data, facet_row, facet_col, title_dim, **subplot_args)
    for (rlab, clab), facet_df in data.groupby([facet_row, facet_col], observed=True):
        ly = layout.loc[(rlab, clab), :]
        ir, ic = ly["row"], ly["col"]
        cur_args = transform_arguments(facet_df, kwargs)
        trace = go.Heatmap(**cur_args)
        fig.add_trace(trace, row=ir + 1, col=ic + 1)
    return fig


def line(error_y_mode=None, **kwargs):
    """Extension of `plotly.express.line` to use error bands."""
    ERROR_MODES = {"bar", "band", "bars", "bands", None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(
            f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}."
        )
    if error_y_mode in {"bar", "bars", None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {"band", "bands"}:
        if "error_y" not in kwargs:
            raise ValueError(
                f"If you provide argument 'error_y_mode' you must also provide 'error_y'."
            )
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg, val in kwargs.items() if arg != "error_y"})
        for data in figure_with_error_bars.data:
            x = list(data["x"])
            y_upper = list(data["y"] + data["error_y"]["array"])
            y_lower = list(
                data["y"] - data["error_y"]["array"]
                if data["error_y"]["arrayminus"] is None
                else data["y"] - data["error_y"]["arrayminus"]
            )
            color = (
                f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace(
                    "((", "("
                )
                .replace("),", ",")
                .replace(" ", "")
            )
            fig.add_trace(
                go.Scatter(
                    x=x + x[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=color,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=data["legendgroup"],
                    xaxis=data["xaxis"],
                    yaxis=data["yaxis"],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data) / 2)):
            reordered_data.append(fig.data[i + int(len(fig.data) / 2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


def plot_behav(
    data,
    fm="frame",
    pos="linpos",
    pos_sgn="linpos_sign",
    trial="trial",
    ax=None,
    **kwargs,
):
    data = data.copy()
    if ax is None:
        ax = plt.gca()
    # plot trial
    tidx = np.diff(data[trial], prepend=0) > 0
    for f in data[fm][tidx]:
        ax.axvline(f, linestyle=":", color="black", linewidth=0.5)
    # plot position
    seg_cmap = kwargs.get(
        "seg_cmap",
        {0: qualitative.D3[7], 1: qualitative.Dark24[0], -1: qualitative.Dark24[1]},
    )
    seg_lmap = kwargs.get(
        "seg_lmap", {0: "idle", 1: "running right", -1: "running left"}
    )
    data["seg_sign"] = np.sign(data[pos_sgn]).fillna(0)
    data["pos_seg"] = unique_seg(data["seg_sign"])
    for _, seg in data.groupby("pos_seg"):
        s = seg["seg_sign"].unique().item()
        sns.lineplot(
            seg, x=fm, y=pos, color=seg_cmap[s], label=seg_lmap[s], ax=ax, linewidth=0.5
        )
