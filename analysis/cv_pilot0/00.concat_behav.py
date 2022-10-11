#%% imports and definitions
import os
import re

import ffmpeg
import pandas as pd
from natsort import natsorted

IN_DPATH = "./data"
IN_SS_CSV = "./log/sessions.csv"
IN_PAT = r"[0-9]+\.avi$"
OUT_PATH = "./intermediate/behav/concat_avi"

os.makedirs(OUT_PATH, exist_ok=True)


#%% concat videos
ss_csv = pd.read_csv(IN_SS_CSV)
ss_csv = ss_csv[ss_csv["analyze"]]

for _, row in ss_csv.iterrows():
    fpath = os.path.join(IN_DPATH, row["data"], "behavcam")
    flist = natsorted(list(filter(lambda fn: re.search(IN_PAT, fn), os.listdir(fpath))))
    if flist:
        print("processing {}".format(fpath))
        streams = [ffmpeg.input(os.path.join(fpath, vid)) for vid in flist]
        ffmpeg.concat(*streams).output(
            os.path.join(OUT_PATH, "{}-{}.avi".format(row["animal"], row["name"])),
            c="libx264",
            crf=0,
        ).global_args("-loglevel", "error").run(overwrite_output=True)
