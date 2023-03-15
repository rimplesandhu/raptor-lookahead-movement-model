""" Commonly used functions """
from typing import Optional, Union, Dict, List
import numpy as np
from numpy import ndarray
# pylint: disable=invalid-name


def subtract_states(
    x0: ndarray,
    x1: ndarray,
    angle_idx: int
) -> ndarray:
    """Subtract two state vectors that include angle"""
    xres = x0 - x1
    try:
        xres[int(angle_idx)] = np.mod(xres[int(angle_idx)], 2.0 * np.pi)
        if xres[int(angle_idx)] > np.pi:
            xres[angle_idx] -= 2. * np.pi
    except:
        print(xres)
    return xres


def state_mean_func(
    list_of_sigmas: List[ndarray],
    list_of_wgts: List[float],
    angle_idx: int
) -> ndarray:
    """Returns means of state vectors with wgts and making sure angles are a
    added properly"""
    y_mvec = sum([iw * iy for iw, iy in zip(list_of_wgts, list_of_sigmas)])
    for ivec, iwgt in zip(list_of_sigmas, list_of_wgts):
        sum_sin = np.sum(np.dot(np.sin(ivec[int(angle_idx)]), iwgt))
        sum_cos = np.sum(np.dot(np.cos(ivec[int(angle_idx)]), iwgt))
        y_mvec[int(angle_idx)] = np.arctan2(sum_sin, sum_cos)
    return y_mvec


def symmetrize(in_mat: ndarray) -> ndarray:
    """Return a symmetrized version of NumPy array"""
    # if np.any(np.isnan(in_mat)) or np.any(in_mat.diagonal() < 0.):
    #     print('\np update went wrong!')
    #     print(in_mat.diagonal())
    return (in_mat + in_mat.T) / 2.


def get_covariance_ellipse(icov: ndarray, fac: float):
    """Retruns width, height, and angle of the covariance ellipse"""
    vals, vecs = np.linalg.eigh(icov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2. * fac * np.sqrt(vals)
    return width, height, theta
