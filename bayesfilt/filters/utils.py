""" Commonly used functions """
import numpy as np
from numpy import ndarray
from copy import deepcopy
# pylint: disable=invalid-name

clrs = ['#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']


def subtract_func(
    x1: ndarray,
    x2: ndarray,
    angle_index: int | None = None,
    radians: bool = False
) -> ndarray:
    """Subtract two vectors that include angle"""
    # x0 = self.matrix(x0, self.nx)
    # x1 = self.matrix(x1, self.nx)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    ival = np.pi if radians else 180.
    assert x1.size == x2.size, 'Shape mismatch!'
    xres = np.subtract(x1, x2)
    if angle_index is not None:
        # angle_index = self.scaler(angle_index, dtype='int32')
        angle_index = int(angle_index)
        assert angle_index < x1.size, 'Invalid angle index!'
        # xres[angle_index] = np.mod(xres[angle_index], 2.0 * np.pi)
        if xres[angle_index] > ival:
            xres[angle_index] -= 2. * ival
        if xres[angle_index] <= -ival:
            xres[angle_index] += 2. * ival
        # xres[angle_index] = angle_correction(xres[angle_index])
    return xres


def angle_correction(iangle):
    """Correct angles to be between -pi,pi"""
    new_angle = iangle
    if iangle > np.pi:
        new_angle = iangle - 2. * np.pi
    elif iangle <= -np.pi:
        new_angle = iangle + 2. * np.pi
    return new_angle


def mean_func(
    list_of_vecs: list[ndarray],
    weights: list[float] | None = None,
    angle_index: int | None = None,
    radians: bool = False
) -> ndarray:
    """Returns means of vectors with wgts while handling angles"""
    assert isinstance(list_of_vecs, list), 'Need a list of numpy arrays'
    if weights is None:
        weights = [1. / len(list_of_vecs)] * len(list_of_vecs)
    if len(list_of_vecs) != len(weights):
        raise ValueError('Size mismatch between vecs and wgts!')
    yvec = sum([iw * iy for iw, iy in zip(weights, list_of_vecs)])
    if angle_index is not None:
        angle_index = int(angle_index)
        angles = np.array([ivec[angle_index] for ivec in list_of_vecs])
        if not radians:
            angles = np.radians(angles)
        # print(np.degrees(angles))
        # print(np.sin(angles))
        # print(np.cos(angles))
        # print(np.multiply(np.sin(angles), weights))
        # print(np.multiply(np.cos(angles), weights))
        # angles = np.vectorize(angle_correction)(angles)
        # print(np.degrees(angles))
        # print(np.sin(angles))
        # print(np.cos(angles))
        sum_sin = np.dot(np.sin(angles), weights)
        sum_cos = np.dot(np.cos(angles), weights)
        yvec[angle_index] = np.arctan2(sum_sin, sum_cos)
        if not radians:
            yvec[angle_index] = np.degrees(yvec[angle_index])
        # print(weights, sum_sin, sum_cos, yvec)
    return yvec


def get_covariance_ellipse(icov: ndarray, fac: float):
    """Retruns width, height, and angle of the covariance ellipse"""
    vals, vecs = np.linalg.eigh(icov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2. * fac * np.sqrt(vals)
    return width, height, theta


def validate_array(
    in_mat: ndarray,
    in_shape: tuple | int | None = None,
    return_array=False,
    error: Exception = ValueError
) -> ndarray:
    """Returns a valid numpy array while checking for its shape and validity"""
    out_mat = None
    in_mat = np.atleast_1d(np.asarray_chkfinite(in_mat, dtype=float))
    if in_shape is not None:
        in_shape = (in_shape,) if isinstance(in_shape, int) else in_shape
        in_shape = (in_shape,) if np.isscalar(in_shape) else in_shape
        if in_mat.shape != in_shape:
            out_str = f'Required shape:{in_shape}, Input shape: {in_mat.shape}'
            raise error(out_str)
    out_mat = in_mat if return_array else out_mat
    return out_mat


def sym_posdef_matrix(in_mat: ndarray) -> ndarray:
    """Return a symmetrized version of NumPy array"""
    if np.any(np.isnan(in_mat)):
        raise ValueError('Matrix has nans!')
    if np.any(in_mat.diagonal() < 0.):
        raise ValueError('Matrix has negative entries on diagonal!')
    try:
        _ = np.linalg.cholesky(in_mat)
    except np.linalg.linalg.LinAlgError as _:
        print('Matrix is not pos def!')
        raise
    return (in_mat + in_mat.T) / 2.
