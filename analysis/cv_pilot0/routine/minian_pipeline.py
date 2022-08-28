import os
import shutil

import dask.array as darr
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.initialization import initA, initC, pnr_refine, seeds_init, seeds_merge
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    TaskAnnotation,
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
)
from minian.visualization import generate_videos

from .alignment import apply_affine


def minian_process(
    dpath,
    intpath,
    param,
    glow_rm=True,
    return_stage=None,
    varr=None,
    client=None,
    n_workers=None,
    flip=False,
    tx=None,
):
    # setup
    dpath = os.path.abspath(os.path.expanduser(dpath))
    intpath = os.path.abspath(os.path.expanduser(intpath))
    shutil.rmtree(intpath, ignore_errors=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    if client is None:
        cluster = LocalCluster(
            n_workers=n_workers,
            memory_limit="4GB",
            resources={"MEM": 1},
            threads_per_worker=2,
            dashboard_address="0.0.0.0:12345",
        )
        annt_plugin = TaskAnnotation()
        cluster.scheduler.add_plugin(annt_plugin)
        client = Client(cluster)
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
    varr_ref = varr.sel(param["subset"])
    if return_stage == "load":
        return varr
    # preprocessing
    if glow_rm:
        varr_min = varr_ref.min("frame").compute()
        varr_ref = varr_ref - varr_min
    varr_ref = denoise(varr_ref, **param["denoise"])
    if param["background_removal"]["method"] == "uniform":
        varr_ref = (
            remove_background(varr_ref.astype(float), **param["background_removal"])
            .clip(0, 255)
            .astype(np.uint8)
        )
    else:
        varr_ref = remove_background(varr_ref, **param["background_removal"])
    if param.get("background_removal_it2"):
        varr_ref = remove_background(varr_ref, **param["background_removal_it2"])
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
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)
    if return_stage == "preprocessing":
        return varr_ref
    # motion-correction
    motion = estimate_motion(varr_ref, **param["estimate_motion"])
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param["save_minian"]
    )
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    if return_stage == "motion-correction":
        return motion, Y_fm_chk, Y_hw_chk
    # initilization
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param["save_minian"]
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
    except KeyError:
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
    if return_stage == "initialization":
        return A, C, b, f
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
        return A, C, b, f
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
        return A, C, S, b, f
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
        return A, C, S, b, f
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
    # save result
    A = save_minian(A.rename("A"), **param["save_minian"])
    C = save_minian(C.rename("C"), **param["save_minian"])
    S = save_minian(S.rename("S"), **param["save_minian"])
    c0 = save_minian(c0.rename("c0"), **param["save_minian"])
    b0 = save_minian(b0.rename("b0"), **param["save_minian"])
    b = save_minian(b.rename("b"), **param["save_minian"])
    f = save_minian(f.rename("f"), **param["save_minian"])
    # generate video
    generate_videos(varr, Y_fm_chk, A=A, C=C_chk, vpath=dpath)
    return A, C, S, b, f
