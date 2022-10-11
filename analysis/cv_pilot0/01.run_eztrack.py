#%% imports and definitions
import os
import shutil

import holoviews as hv
import numpy as np

from routine.eztrack import Batch_LoadFiles, Batch_Process

IN_DPATH = "./intermediate/behav/concat_avi"
IN_MASK = "./intermediate/behav/eztrack_mask.npy"
OUT_PATH = "./intermediate/behav/eztrack"
FIG_PATH = "./figs/eztrack"

video_dict = {
    "dpath": IN_DPATH,
    "ftype": "avi",
    "start": 0,
    "end": None,
    "region_names": None,
    "dsmpl": 1,
    "stretch": dict(width=1, height=1),
}
tracking_params = {
    "loc_thresh": 99,
    "use_window": True,
    "window_size": 100,
    "window_weight": 0.9,
    "method": "abs",
    "rmv_wire": False,
    "wire_krn": 10,
}
bin_dict = None

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)
#%% load data
video_dict = Batch_LoadFiles(video_dict)
video_dict["mask"] = {"mask": np.load(IN_MASK)}

#%% process
summary, images = Batch_Process(video_dict, tracking_params, bin_dict)
hv.save(images.cols(4), os.path.join(FIG_PATH, "master.html"))
os.remove(os.path.join(IN_DPATH, "BatchSummary.csv"))
for file in os.listdir(IN_DPATH):
    if file.endswith(".csv"):
        shutil.move(
            os.path.join(IN_DPATH, file),
            os.path.join(OUT_PATH, file.replace("_LocationOutput", "")),
        )
