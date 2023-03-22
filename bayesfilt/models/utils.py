""" Commonly used functions """
# pylint: disable=invalid-name
# from typing import Optional, Union, Dict, List
import numpy as np
from numpy import ndarray


def get_covariance_ellipse(icov: ndarray, fac: float):
    """Retruns width, height, and angle of the covariance ellipse"""
    vals, vecs = np.linalg.eigh(icov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2. * fac * np.sqrt(vals)
    return width, height, theta
