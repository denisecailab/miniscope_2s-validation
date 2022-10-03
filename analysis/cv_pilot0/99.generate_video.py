#%% imports and definition
import os
import shutil

import colorcet as cc
import cv2
import ffmpeg
import holoviews as hv
import numpy as np
import xarray as xr
import yaml
from distributed import Client, LocalCluster
from matplotlib import cm
from minian.cnmf import compute_AtC
from minian.utilities import TaskAnnotation, load_videos, rechunk_like, save_minian

from routine.alignment import apply_affine, est_affine
from routine.minian_pipeline import minian_process
from routine.utilities import norm_xr

hv.notebook_extension("bokeh")

PARAM_SIDE = "./process_parameters/green_channel/generic.yaml"
PARAM_TOP = "./process_parameters/red_channel/generic.yaml"
IN_SIDE_PATH = "./intermediate/processed/green/m15-rec1.nc"
IN_TOP_PATH = "./intermediate/processed/red/m15-rec1.nc"
IN_DSPATH = "./data/m15/2022_08_04/12_34_43"
INTPATH = "~/var/miniscope2s-validation/minian_int"
INTPATH_SIDE = "~/var/miniscope2s-validation/minian_int-green"
INTPATH_TOP = "~/var/miniscope2s-validation/minian_int-red"
IN_WORKER_PATH = "~/var/miniscope2s-validation/dask-worker-space"
INTPATH = os.path.abspath(os.path.expanduser(INTPATH))
INTPATH_SIDE = os.path.abspath(os.path.expanduser(INTPATH_SIDE))
INTPATH_TOP = os.path.abspath(os.path.expanduser(INTPATH_TOP))
IN_WORKER_PATH = os.path.abspath(os.path.expanduser(IN_WORKER_PATH))
SUBSET = {"frame": slice(6000, 7999, 2)}
SUBSET_BEHAV = {
    "frame": slice(6000, 7999, 2),
    "height": slice(215, 335),
    "width": slice(90, 530),
}
ROT_BEHAV = -5.5
ANNT_COL = (255, 255, 255)
CLIP = (5, 25)
BRT_OFFSET = 0
OUTPATH = "./output/video/m15"
TITLES = {
    "top": "tdTomato",
    "side": "GCamp",
    "ovly": "Overlay",
}
os.makedirs(OUTPATH, exist_ok=True)


def annt(fm, text):
    return cv2.putText(
        (fm * 255).astype(np.uint8),
        text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        ANNT_COL,
        2,
    )


def make_combined_frame(gfm, gfm_ps, rfm, rfm_ps, behav=None):
    gcm = cm.ScalarMappable(cmap=cc.m_linear_ternary_green_0_46_c42)
    rcm = cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52)
    gfm, gfm_ps = (
        gcm.to_rgba(gfm, norm=False)[:, :, :3],
        gcm.to_rgba(gfm_ps, norm=False)[:, :, :3],
    )
    rfm, rfm_ps = (
        rcm.to_rgba(rfm, norm=False)[:, :, :3],
        rcm.to_rgba(rfm_ps, norm=False)[:, :, :3],
    )
    ovlp, ovlp_ps = np.clip(gfm + rfm * 0.75, 0, 1), np.clip(gfm_ps + rfm_ps, 0, 1)
    gfm, gfm_ps, rfm, rfm_ps, ovlp, ovlp_ps = (
        annt(gfm, "GCaMP"),
        annt(gfm_ps, "GCaMP processed"),
        annt(rfm, "tdTomato"),
        annt(rfm_ps, "tdTomato processed"),
        annt(ovlp, "Overlap"),
        annt(ovlp_ps, "Overlap processed"),
    )
    out_im = np.concatenate(
        [
            np.concatenate([gfm, rfm, ovlp], axis=1),
            np.concatenate([gfm_ps, rfm_ps, ovlp_ps], axis=1),
        ],
        axis=0,
    )
    if behav is not None:
        w = out_im.shape[1]
        behav = (
            cm.ScalarMappable(cmap=cc.m_gray)
            .to_rgba(behav, norm=False, bytes=True)[:, :, :3]
            .astype(np.uint8)
        )
        behav = cv2.resize(behav, dsize=(w, int(w / behav.shape[1] * behav.shape[0])))
        out_im = np.concatenate([behav, out_im], axis=0)
    return out_im.astype(np.uint8)


def write_video(
    arr: xr.DataArray,
    vname: str = None,
    options={"crf": "18", "preset": "slow"},
) -> str:
    w, h = arr.sizes["width"], arr.sizes["height"]
    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(w, h), r=60
        )
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        .output(vname, pix_fmt="yuv420p", vcodec="libx264", r=60, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return vname


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


if __name__ == "__main__":
    # load and process calcium data
    cluster = LocalCluster(
        n_workers=8,
        memory_limit="10GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
        local_directory=IN_WORKER_PATH,
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    shutil.rmtree(INTPATH_TOP, ignore_errors=True)
    shutil.rmtree(INTPATH_SIDE, ignore_errors=True)
    side_ds = (
        xr.open_dataset(IN_SIDE_PATH)
        .squeeze()
        .drop(["animal", "session"])
        .sel(**SUBSET)
    )
    top_ds = (
        xr.open_dataset(IN_TOP_PATH).squeeze().drop(["animal", "session"]).sel(**SUBSET)
    )
    fm_side = side_ds["max_proj"].squeeze()
    fm_top = top_ds["max_proj"].squeeze()
    motion = top_ds["motion"].compute()
    tx, param_df = est_affine(fm_side.values, fm_top.values, lr=1)
    with open(PARAM_SIDE) as yf:
        param_side = yaml.full_load(yf)
    with open(PARAM_TOP) as yf:
        param_top = yaml.full_load(yf)
    param_side["subset"] = SUBSET
    param_top["subset"] = SUBSET
    intds_top, plots_top = minian_process(
        os.path.join(IN_DSPATH, "miniscope_top"),
        INTPATH_TOP,
        param_top,
        return_stage="motion-correction",
        motion=motion,
    )
    intds_side, plots_side = minian_process(
        os.path.join(IN_DSPATH, "miniscope_side"),
        INTPATH_SIDE,
        param_side,
        return_stage="motion-correction",
        flip=True,
        tx=tx,
        motion=motion,
    )
    # load and process behavior
    shutil.rmtree(INTPATH, ignore_errors=True)
    behav_vid = load_videos(
        os.path.join(IN_DSPATH, "behavcam"), pattern=r"[0-9]+\.avi$"
    )
    behav_vid = xr.apply_ufunc(
        rotate_image,
        behav_vid,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"angle": ROT_BEHAV},
    )
    behav_vid = save_minian(
        behav_vid.sel(**SUBSET_BEHAV).rename("behav_vid"), INTPATH, overwrite=True
    )
    # compute and normalization
    gRaw = intds_side["Y_fm_chk"]
    rRaw = intds_top["Y_fm_chk"]
    chk = {d: c for d, c in zip(gRaw.dims, gRaw.chunks)}
    gAC = compute_AtC(side_ds["A"].chunk(), side_ds["C"].chunk({"frame": chk["frame"]}))
    _, rAC = xr.broadcast(gAC, top_ds["A"].max("unit_id").compute().chunk())
    rAC = rechunk_like(rAC, gAC)
    gAC = xr.apply_ufunc(
        apply_affine,
        gAC,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        kwargs={"tx": tx},
        vectorize=True,
        dask="parallelized",
    )
    gAC = save_minian(gAC.rename("gAC"), INTPATH, overwrite=True)
    rAC = save_minian(rAC.rename("rAC"), INTPATH, overwrite=True)
    gRaw, rRaw, gAC, rAC = (
        norm_xr(gRaw, q=0.999),
        norm_xr(rRaw, q=0.98),
        norm_xr(gAC, q=0.999),
        norm_xr(rAC, q=0.999),
    )
    behav_vid = norm_xr(behav_vid)
    # generate video
    test_fm = make_combined_frame(
        gRaw.isel(frame=0).values,
        gAC.isel(frame=0).values,
        rRaw.isel(frame=0).values,
        rAC.isel(frame=0).values,
        behav_vid.isel(frame=0).values,
    )
    out = xr.apply_ufunc(
        make_combined_frame,
        gRaw,
        gAC,
        rRaw,
        rAC,
        behav_vid.rename({"height": "height_b", "width": "width_b"}),
        input_core_dims=[["height", "width"]] * 4 + [["height_b", "width_b"]],
        output_core_dims=[["height_new", "width_new", "rgb"]],
        vectorize=True,
        output_sizes={
            "height_new": test_fm.shape[0],
            "width_new": test_fm.shape[1],
            "rgb": test_fm.shape[2],
        },
        dask="parallelized",
    )
    out = out.rename({"height_new": "height", "width_new": "width"})
    os.makedirs(OUTPATH, exist_ok=True)
    write_video(out, os.path.join(OUTPATH, "combined.mp4"))
