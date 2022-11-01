import re

import numpy as np
import pandas as pd
import SimpleITK as sitk
from minian.motion_correction import est_motion_perframe


def apply_affine(fm: np.ndarray, tx: sitk.Transform, fill: float = 0):
    fm = sitk.GetImageFromArray(fm)
    fm = sitk.Resample(fm, fm, tx, sitk.sitkLinear, fill)
    return sitk.GetArrayFromImage(fm)


def it_callback(reg, param_dict):
    param = reg.GetOptimizerPosition()
    param_dict[param] = reg.GetMetricValue()


def est_affine(
    src: np.ndarray,
    dst: np.ndarray,
    src_ma=None,
    dst_ma=None,
    lr: float = 0.5,
    niter: int = 1000,
    typ="affine",
):
    src, dst = np.array(src), np.array(dst)
    sh = est_motion_perframe(src, dst, upsample=100)
    src = sitk.GetImageFromArray(src.astype(np.float32))
    dst = sitk.GetImageFromArray(dst.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    if src_ma is not None:
        reg.SetMetricMovingMask(sitk.GetImageFromArray(src_ma.astype(np.uint8)))
    if dst_ma is not None:
        reg.SetMetricFixedMask(sitk.GetImageFromArray(dst_ma.astype(np.uint8)))
    if typ == "affine":
        topt = sitk.AffineTransform(2)
    elif typ == "euler":
        topt = sitk.Euler2DTransform()
    elif typ == "translation":
        return sitk.TranslationTransform(2, (-sh[1], -sh[0])), None
    else:
        raise ValueError("Don't understand transform: {}".format(typ))
    trans_opt = sitk.CenteredTransformInitializer(
        dst,
        src,
        topt,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    trans_opt.SetParameters(np.array([1.0, 0.0, 0.0, 1.0, -sh[1], -sh[0]]))
    # reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SetInitialTransform(trans_opt)
    reg.SetMetricAsMeanSquares()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=lr,
        minStep=1e-5,
        numberOfIterations=niter,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    param_dict = dict()
    reg.AddCommand(sitk.sitkIterationEvent, lambda: it_callback(reg, param_dict))
    tx = reg.Execute(dst, src).Downcast()
    param_df = (
        pd.Series(param_dict)
        .reset_index(name="metric")
        .rename(lambda c: re.sub("level_", "param_", c), axis="columns")
    )
    return tx, param_df
