import os
import shutil

import dask.array as darr
import holoviews as hv
import numpy as np
import xarray as xr
from holoviews import opts
from minian.cnmf import (compute_trace, get_noise_fft, unit_merge,
                         update_background, update_spatial, update_temporal)
from minian.initialization import (initA, initC, pnr_refine, seeds_init,
                                   seeds_merge)
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import get_optimal_chk, load_videos, save_minian
from minian.visualization import (generate_videos, visualize_motion,
                                  visualize_seeds)

from .alignment import apply_affine
from .plotting import plotA_contour
from .static_channel import constructA, find_seed, mergeA
from .stripe_correction import label_good_frames
from .utilities import resample_motion


def minian_process(
    dpath,
    intpath,
    param,
    return_stage=None,
    varr=None,
    flip=False,
    tx=None,
    video_path=None,
    motion=None,
):
    # setup
    dpath = os.path.abspath(os.path.expanduser(dpath))
    intpath = os.path.abspath(os.path.expanduser(intpath))
    shutil.rmtree(intpath, ignore_errors=True)
    # plotting options
    plots = dict()
    opts_crv = opts.Curve(**{"frame_width": 700, "aspect": 2})
    opts_tr = opts.Image(**{"frame_width": 700, "aspect": 2, "cmap": "viridis"})
    opts_im = opts.Image(
        **{
            "frame_width": 400,
            "aspect": 1,
            "cmap": "viridis",
            "colorbar": True,
        }
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    if varr is None:
        varr = load_videos(dpath, **param["load_videos"])
    else:
        del varr.encoding["chunks"]
    chk, _ = get_optimal_chk(varr, dtype=float)
    if flip:
        varr = xr.apply_ufunc(
            darr.flip,
            varr,
            input_core_dims=[["frame", "height", "width"]],
            output_core_dims=[["frame", "height", "width"]],
            kwargs={"axis": 1},
            dask="allowed",
        )
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    if param["stripe_corr"] is not None:
        good_frame = label_good_frames(varr, **param["stripe_corr"])
        varr = varr.sel(frame=good_frame)
    varr_ref = varr.sel(param.get("subset"))
    # preprocessing
    if param["glow_rm"] == "min":
        varr_min = varr_ref.min("frame").compute()
        varr_ref = varr_ref - varr_min
    else:
        varr_ref = (
            remove_background(varr_ref.astype(float), **param["glow_rm"])
            .clip(0, 255)
            .astype(np.uint8)
        )
    varr_ref = denoise(varr_ref, **param["denoise"])
    varr_ref = remove_background(varr_ref, **param["background_removal"])
    if tx is not None:
        varr_ref = xr.apply_ufunc(
            apply_affine,
            varr_ref,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            kwargs={"tx": tx},
            vectorize=True,
            dask="parallelized",
        )
    varr_ref = save_minian(
        varr_ref.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename(
            "varr_ref"
        ),
        dpath=intpath,
        overwrite=True,
    )
    if return_stage == "pre-processing":
        return varr_ref, plots
    # motion-correction
    if motion is None:
        motion = estimate_motion(varr_ref, **param["estimate_motion"])
    else:
        motion = xr.DataArray(
            resample_motion(
                motion.values,
                f_org=motion.coords["frame"].values,
                f_new=varr_ref.coords["frame"].values,
            ),
            dims=motion.dims,
            coords={
                "frame": varr_ref.coords["frame"].values,
                "shift_dim": motion.coords["shift_dim"].values,
            },
        ).sel(frame=varr_ref.coords["frame"].values)
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), intpath, overwrite=True
    )
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    plots["motion"] = (
        hv.Image(
            varr_ref.max("frame").compute().astype(np.float32),
            ["width", "height"],
            label="before_mc",
        ).opts(opts_im)
        + hv.Image(
            Y_hw_chk.max("frame").compute().astype(np.float32),
            ["width", "height"],
            label="after_mc",
        ).opts(opts_im)
        + visualize_motion(motion.compute()).opts(opts_crv)
    ).cols(2)
    if return_stage == "motion-correction":
        return xr.merge([motion, Y_fm_chk, Y_hw_chk]), plots
    # initilization
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), intpath, overwrite=True
    ).compute()
    seeds = seeds_init(Y_fm_chk, **param["seeds_init"])
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param["pnr_refine"])
    seeds_final = seeds[seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param["seeds_merge"])
    A_init = initA(
        Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param["initialize"]
    )
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    try:
        A, C = unit_merge(A_init, C_init, **param["init_merge"])
    except (KeyError, ValueError):
        A, C = A_init, C_init
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)
    plots["init"] = (
        hv.Image(
            A.max("unit_id").rename("A").compute().astype(np.float32),
            kdims=["width", "height"],
        )
        .opts(opts_im)
        .relabel("Initial Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).rename("C").compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Initial Temporal Components")
        + hv.Image(
            b.rename("b").compute().astype(np.float32), kdims=["width", "height"]
        )
        .opts(opts_im)
        .relabel("Initial Background Sptial")
        + hv.Curve(f.rename("f").compute(), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Initial Background Temporal")
    ).cols(2)
    if return_stage == "initialization":
        return xr.merge([A, C, b, f]), plots
    # cnmf
    sn_spatial = get_noise_fft(Y_hw_chk, **param["get_noise"])
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
    ## first iteration
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, **param["first_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    if return_stage == "first-spatial":
        return xr.merge([A, C, b, f]), plots
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param["first_temporal"]
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)
    if return_stage == "first-temporal":
        return xr.merge([A, C, S, b, f]), plots
    ## merge
    try:
        A_mrg, C_mrg, [sig_mrg] = unit_merge(
            A, C, [C + b0 + c0], **param["first_merge"]
        )
    except KeyError:
        A_mrg, C_mrg, sig_mrg = A, C, C + b0 + c0
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)
    ## second iteration
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, sig, sn_spatial, **param["second_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    if return_stage == "second-spatial":
        return xr.merge([A, C, S, b, f]), plots
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param["second_temporal"]
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)
    plots["final"] = (
        hv.Image(
            A.max("unit_id").compute().astype(np.float32),
            kdims=["width", "height"],
        )
        .opts(opts_im)
        .relabel("Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Temporal Trace")
        + hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"])
        .opts(opts_im)
        .relabel("Background Spatial")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Background Temporal")
    ).cols(2)
    # generate video
    if video_path is not None:
        vpath, vname = os.path.split(video_path)
        generate_videos(varr, Y_fm_chk, A=A, C=C_chk, vpath=vpath, vname=vname)
    result_ds = xr.merge(
        [
            A.rename("A"),
            C.rename("C"),
            YrA.rename("YrA"),
            S.rename("S"),
            c0.rename("c0"),
            b0.rename("b0"),
            b.rename("b"),
            f.rename("f"),
            motion,
            max_proj,
        ]
    )
    return result_ds, plots


def red_channel(dpath, intpath, param):
    opts_im = opts.Image(
        **{
            "frame_width": 400,
            "aspect": 1,
            "cmap": "viridis",
            "colorbar": True,
        }
    )
    result_ds, plots = minian_process(
        dpath, intpath, param, return_stage="motion-correction"
    )
    Y = result_ds["Y_fm_chk"]
    max_proj = save_minian(
        Y.mean("frame").rename("max_proj"), intpath, overwrite=True
    ).compute()
    seeds = find_seed(max_proj, **param["find_seed"])
    A = constructA(seeds, max_proj, **param["constructA"])
    A_merge = mergeA(A, **param["mergeA"]).rename("A")
    C = initC(Y, A_merge.chunk({"height": -1, "width": -1, "unit_id": 1}))
    C = save_minian(
        C.rename("C_init"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    plots["footprints"] = (
        visualize_seeds(max_proj, seeds).opts(opts_im)
        + plotA_contour(A_merge, max_proj).opts(opts_im)
        + hv.Image(A_merge.max("unit_id"), ["width", "height"]).opts(opts_im)
    )
    return xr.merge([result_ds["motion"], max_proj, A_merge, C]), plots
