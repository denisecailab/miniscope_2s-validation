"""
script to run minian pipeline on datasets

env: environments/minian.yml
"""

import io
import os
import shutil
import traceback
import warnings
from contextlib import redirect_stdout
from datetime import datetime

import dask as da
import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from bokeh.plotting import save
from bokeh.resources import CDN
from distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation

from routine.minian_pipeline import minian_process
from routine.utilities import subset_sessions

IN_DPATH = "./data"
IN_MINIAN_INT_PATH = "~/var/miniscope2s-validation/minian_int"
IN_WORKER_PATH = "~/var/miniscope2s-validation/dask-worker-space"
IN_PARAM_FOLDER = "./process_parameters/green_channel"
IN_RED_PATH = "./intermediate/processed/red"
IN_SSMAP = "log/sessions.csv"
IN_SS_SUB = r".*"
IN_ANM_SUB = r".*"
PARAM_SKIP_EXSISTING = True
OUT_ERR_FOLDER = "./process_error"
OUT_VID_PATH = "./intermediate/green_video"
OUT_PATH = "./intermediate/processed/green"
FIG_FOLDER = "./figs/processed/green_channel"
os.makedirs(OUT_VID_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(OUT_ERR_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


if __name__ == "__main__":
    # setup
    IN_MINIAN_INT_PATH = os.path.abspath(os.path.expanduser(IN_MINIAN_INT_PATH))
    IN_WORKER_PATH = os.path.abspath(os.path.expanduser(IN_WORKER_PATH))
    IN_DPATH = os.path.abspath(IN_DPATH)
    hv.extension("bokeh")
    hv.config.image_rtol = 100
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = IN_MINIAN_INT_PATH
    np.seterr(all="ignore")
    da.config.set(
        **{
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
        }
    )
    # read session map
    ssmap = pd.read_csv(IN_SSMAP)
    ssmap = ssmap[ssmap["data"].notnull()]
    ssmap = ssmap[
        ssmap.apply(
            subset_sessions, anm_pat=IN_ANM_SUB, ss_pat=IN_SS_SUB, axis="columns"
        )
        & ssmap["analyze"]
    ]
    # process loop
    for _, ss_row in ssmap.iterrows():
        dp, anm, ss = ss_row["data"], ss_row["animal"], ss_row["name"]
        if PARAM_SKIP_EXSISTING and os.path.exists(
            os.path.join(OUT_PATH, "{}-{}.nc".format(anm, ss))
        ):
            print("skipping {}".format(dp))
            continue
        # determine parameters
        with open(os.path.join(IN_PARAM_FOLDER, "generic.yaml")) as yf:
            param = yaml.full_load(yf)
        try:
            with open(os.path.join(IN_PARAM_FOLDER, "{}.yaml".format(anm))) as yf:
                param_anm = yaml.full_load(yf)
            param.update(param_anm)
            print("using config {}.yaml".format(anm))
        except FileNotFoundError:
            pass
        try:
            with open(
                os.path.join(IN_PARAM_FOLDER, "{}-{}.yaml".format(anm, ss))
            ) as yf:
                param_ss = yaml.full_load(yf)
            param.update(param_ss)
            print("using config {}-{}.yaml".format(anm, ss))
        except FileNotFoundError:
            pass
        try:
            motion = (
                xr.open_dataset(os.path.join(IN_RED_PATH, "{}-{}.nc".format(anm, ss)))[
                    "motion"
                ]
                .squeeze()
                .drop(["animal", "session"])
                .compute()
            )
        except FileNotFoundError:
            motion = None
            warnings.warn("Can't load motion for {} {}".format(anm, ss))
        shutil.rmtree(IN_MINIAN_INT_PATH, ignore_errors=True)
        # start cluster
        started = False
        while not started:
            try:
                cluster = LocalCluster(
                    n_workers=12,
                    memory_limit="6GB",
                    resources={"MEM": 1},
                    threads_per_worker=2,
                    dashboard_address="0.0.0.0:12345",
                    local_directory=IN_WORKER_PATH,
                )
                annt_plugin = TaskAnnotation()
                cluster.scheduler.add_plugin(annt_plugin)
                client = Client(cluster)
                started = True
            except:
                cluster.close()
        try:
            tstart = datetime.now()
            with redirect_stdout(io.StringIO()):
                result_ds, plots = minian_process(
                    dpath=os.path.join(IN_DPATH, dp, "miniscope_side"),
                    intpath=IN_MINIAN_INT_PATH,
                    param=param,
                    flip=True,
                    motion=motion,
                )
            tend = datetime.now()
            print("minian success: {}".format(dp))
            print("time: {}".format(tend - tstart))
        except Exception as err:
            print("minian failed: {}".format(dp))
            with open(os.path.join(OUT_ERR_FOLDER, "-".join([anm, ss])), "w") as txtf:
                traceback.print_exception(None, err, err.__traceback__, file=txtf)
            client.close()
            cluster.close()
            continue
        result_ds = (
            result_ds.assign_coords(animal=anm, session=ss)
            .expand_dims(["animal", "session"])
            .compute()
        )
        result_ds.to_netcdf(
            os.path.join(OUT_PATH, "{}-{}.nc".format(anm, ss)), format="NETCDF4"
        )
        for plt_name, plt in plots.items():
            plt_name = "-".join([anm, ss, plt_name])
            save(
                hv.render(plt),
                os.path.join(FIG_FOLDER, "{}.html".format(plt_name)),
                title=plt_name,
                resources=CDN,
            )
        client.close()
        cluster.close()
