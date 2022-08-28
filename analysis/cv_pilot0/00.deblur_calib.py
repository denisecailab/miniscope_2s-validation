#%% imports and definitions
import os

import holoviews as hv
import matplotlib.pyplot as plt
import torch
from minian.preprocessing import remove_background_perframe
from minian.utilities import load_videos
from skimage import io
from skimage.transform import resize

from routine.deblurring.functions import calibrate, deblur
from routine.utilities import norm

hv.notebook_extension("bokeh")

IN_DPATH = "./data/bench/2022_07_23/20_03_20"
FIG_PATH = "./figs/deblur"
os.makedirs(FIG_PATH, exist_ok=True)

#%% load data and compute calibration image
vside = load_videos(os.path.join(IN_DPATH, "miniscope_side"), pattern=r"[0-9]+\.avi")
vtop = load_videos(os.path.join(IN_DPATH, "miniscope_top"), pattern=r"[0-9]+\.avi")
im_side = vside.median("frame").compute()
im_top = vtop.median("frame").compute()
im_side = norm(
    remove_background_perframe(im_side.values, method="uniform", wnd=50, selem=None)
)
im_top = norm(
    remove_background_perframe(im_top.values, method="uniform", wnd=50, selem=None)
)

#%% subset
im_side = im_side[150:450, 150:450]
im_top = im_top[150:450, 150:450]

#%% calibrate
dim = 300
num_psfs = 300
opt_params = {"iters": 300, "lr": 7.5e-3, "reg": 0}
device = torch.device("cpu")
# psf_stack_side, seidel_side = calibrate(
#     im_side.copy(),
#     seidel_coeffs=None,
#     desired_dim=dim,
#     num_psfs=num_psfs,
#     opt_params=opt_params,
#     device=device,
#     centered_psf=False,
# )
psf_stack_top, seidel_top = calibrate(
    im_top.copy(),
    seidel_coeffs=None,
    desired_dim=dim,
    num_psfs=num_psfs,
    opt_params=opt_params,
    device=device,
    centered_psf=False,
)

#%% deblur
opt_params = {
    "iters": 100,
    "optimizer": "adam",
    "lr": 7.5e-2,
    "init": "measurement",
    "crop": 0,
    "reg": 7.5e-11,
}
# deblurred_side = deblur(
#     im_side.copy(),
#     psf_stack_side,
#     opt_params=opt_params,
#     artifact_correction=0.2,
#     device=device,
# )
deblurred_top = deblur(
    im_top.copy(),
    psf_stack_top,
    opt_params=opt_params,
    artifact_correction=0.2,
    device=device,
)

#%% plotting
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(norm(im_top))
ax1.imshow(norm(deblurred_top))
ax0.axis("off")
ax1.axis("off")
ax0.set_title("calibration image")
ax1.set_title("deblurred")
fig.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "deblur.png"), dpi=500)
# opts = {"cmap": "viridis"}
# hv.Image(norm(im_side), name="before").opts(**opts) + hv.Image(norm(deblurred_side), name="after").opts(**opts)
# hv.Image(norm(im_top), name="before").opts(**opts) + hv.Image(
#     norm(deblurred_top), name="after"
# ).opts(**opts)
