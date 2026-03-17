"""
A module with some usefull utility functions.

(c) 2026 Pablo Krupa
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from numpy.typing import ArrayLike
from typing import Any
import copy as cp

# =============================================================================
# Functions related to working with matrices
# =============================================================================


@jax.jit
def triu2array(mat: ArrayLike) -> ArrayLike:
    """ Returns an array containing the elements of the given upper triangular matrix. """
    return mat[jnp.triu_indices_from(mat)]


@jax.jit
def tril2array(mat: ArrayLike) -> ArrayLike:
    """ Returns an array containing the elements of the given lower triangular matrix. """
    return mat[jnp.tril_indices_from(mat)]


@partial(jax.jit, static_argnames=['dim'])
def array2triu(vec: ArrayLike, dim: int) -> ArrayLike:
    """
    Returns the upper triangular matrix whose elements are given by vec, where
    the ordering is given by jnp.triu_indices().
    The function assumes that the matrix is square and of dimension (dim x dim).
    """
    mat = jnp.empty((dim, dim), vec.dtype)
    mat = mat.at[jnp.triu_indices(dim, k=0)].set(vec)
    return mat


@partial(jax.jit, static_argnames=['dim'])
def array2tril(vec: ArrayLike, dim: int) -> ArrayLike:
    """
    Returns the lower triangular matrix whose elements are given by vec, where
    the ordering is given by jnp.tril_indices().
    The function assumes that the matrix is square and of dimension (dim x dim).
    """
    mat = jnp.empty((dim, dim), vec.dtype)
    mat = mat.at[jnp.tril_indices(dim, k=0)].set(vec)
    return mat


def nnz_chol(dim: int) -> int:
    """Returns number of non-zero elements in the Cholesky decomposition of a positive definite matrix of dimension `dim`"""
    return int((dim * (dim - 1)) / 2.) + dim


def gen_rand_pd_matrix(dim: int,
                       lb_diag: float | np.ndarray = 0.0,
                       ub_diag: float | np.ndarray = 1.0,
                       nondiag_gain: float = 0.0) -> ArrayLike:
    """
    Generates a random positive definite matrix.

    Parameters
    ----------
    dim: int
        Dimention of the matrix
    lb_diag: float, optional
        Indirectly determines the lover bound of the diagonal elements
    ub_diag: float, optional
        Indirectly determines the upper bound of the diagonal elements
    nondiag_gain: float, optional
        Indirectly determines the magnitude of non-diagonal elements

    Returns
    -------
    A numpy.ndarray containing a random positive definite matrix of dimension (dim, dim)
    """
    assert dim > 0 and nondiag_gain >= 0.0
    if isinstance(lb_diag, float):
        assert lb_diag > 0
    else:
        assert all(lb_diag > 0.0)
    if isinstance(ub_diag, float):
        assert ub_diag > 0
    else:
        assert all(ub_diag > 0.0)
    M = nondiag_gain * np.random.randn(dim, dim)  # Take elements from Gaussian distribution, with magnitude nondiag_gain
    M = (M.T @ M) / 2.0  # Ensure it is positive definite
    M = M + np.diag(np.random.uniform(lb_diag, ub_diag, (dim,)))  # Add diagonal term bounded by lb_diag and ub_diag
    return M


# =============================================================================
# Functions related to operations with dictionaries
# =============================================================================


def update_dict(init_dict: dict | Any, add_dict: dict | Any, copy: bool = False):
    """
    Updates the keys in init_dict with the values in add_dict.
    Ignores keys in add_dict that are not in init_dict. Works with nested dicts.

    Parameters
    ----------
    init_dict: dict
        The initial dictionary whose fields will be updated
    add_dict: dict
        The dictionary with which to update the fields of init_dict
    copy: bool, optional
        If set to True, then a copy of init_dict is returned (init_dict is not modified)

    Returns
    -------
    New dict with updated fields. init_dict is also modified if copy=False
    """
    if copy:
        new_dict = cp.deepcopy(init_dict)
    else:
        new_dict = init_dict
    if isinstance(new_dict, dict):
        for k, v in new_dict.items():
            if isinstance(add_dict, dict):
                new_dict[k] = update_dict(new_dict[k], add_dict.get(k, new_dict[k]))
            else:
                new_dict[k] = add_dict
    else:
        new_dict = add_dict
    return new_dict


def update_add_dict(init_dict: dict | Any, add_dict: dict | Any, copy: bool = False):
    """
    Like update_dict(), but also adds the keys in add_dict that are not in init_dict.
    Works like dict.update(), but for nested dicts.

    Parameters
    ----------
    init_dict: dict
        The initial dictionary whose fields will be updated
    add_dict: dict
        The dictionary with which to update the fields of init_dict
    copy: bool, optional
        If set to True, then a copy of init_dict is returned (init_dict is not modified)

    Returns
    -------
    New dict with updated fields. init_dict is also modified if copy=False
    """
    if copy:
        new_dict = cp.deepcopy(init_dict)
    else:
        new_dict = init_dict
    for k, v in add_dict.items():
        if isinstance(v, dict):
            new_dict[k] = update_add_dict(new_dict.get(k, {}), v)
        else:
            new_dict[k] = v
    return new_dict


def dict_from_nested_dict(in_dict: dict, value: Any = None):
    """
    Returns a dict that has the same nested structure (and keys) as the given dict.
    Sets the value of all keys to the given value (defaults to None).
    Similar to creating a dict using dict.fromkeys(), but for nested dicts.

    Parameters
    ----------
    in_dict: dict
        The dict whose structure is copied
    value: Any, optional (defaults to None)
        The value assigned to all the fields of the new dict

    Returns
    -------
    New dict
    """
    new_dict = cp.deepcopy(in_dict)

    def loop_nested_layers(layer):
        for k in layer.keys():
            if not isinstance(layer[k], dict):
                layer[k] = np.full_like(layer[k], value)
            else:
                layer[k] = loop_nested_layers(layer[k])
        return layer

    return loop_nested_layers(new_dict)


def dict_substitute_None(in_dict: dict, value: Any):
    """
    Substitutes all None values in a nested dict with the given value

    Parameters
    ----------
    in_dict: dict
        The dict whose values equal to None will be substuted
    value: Any
        The value assigned to all the None fields in the given dict
    """
    def loop_nested_layers(layer):
        for k in layer.keys():
            if not isinstance(layer[k], dict):
                if layer[k] is None:
                    layer[k] = value
            else:
                loop_nested_layers(layer[k])
    loop_nested_layers(in_dict)


def clip_dict(in_dict: dict, lb: dict | float = -np.inf, ub: dict | float = np.inf, clip_slack: float = 0.0,
              ignore_missing_keys: bool = False):
    """
    Clips the fields in in_dict using the fields in lb and ub. Can handle nested dicts

    Parameters
    ----------
    in_dict: dict
        The dict whose values will be clipped using the lower and upper bound `lb` and `ub`
    lb: dict | float, optional (defaults to -inf)
        Lower bound for clipping in_dict. If it is a dict, it must have the same fields as in_dict
    ub: dict | float, optional (defaults to -inf)
        Upper bound for clipping in_dict. If it is a dict, it must have the same fields as in_dict
    clip_slack: float, optional (defaults to 0)
        A bias used when clipping, as in lb[k] + clip_slack <= in_dict[k] <= ub[k] - clip_slack
    ignore_missing_keys: bool, optional (defaults to False)
        If True, then keys that are in in_dict but not in lb or ub are ignored
    """
    # Convert bounds to dict if a non-dict value is given
    if not isinstance(lb, dict):
        lb = dict_from_nested_dict(in_dict, lb)
    if not isinstance(ub, dict):
        ub = dict_from_nested_dict(in_dict, ub)

    def clip_layer(layer, lb_layer, ub_layer):
        for k in layer.keys():
            if not isinstance(layer[k], dict):
                if ignore_missing_keys and (k not in lb_layer or k not in ub_layer):
                    continue
                layer[k] = np.clip(layer[k], lb_layer[k] + clip_slack, ub_layer[k] - clip_slack)
            else:
                clip_layer(layer[k], lb_layer[k], ub_layer[k])

    clip_layer(in_dict, lb, ub)


def count_size_nested_dict(in_dict: dict) -> int:
    """
    Returns the number of elements in a nested dict.
    Values in the dict are assumed to be: dict, jnp.ndarray, np.ndarray or scalars (float/int)
    """

    def loop_nested_layers(layer, count=0):
        for k in layer.keys():
            if not isinstance(layer[k], dict):
                if isinstance(layer[k], (jnp.ndarray, np.ndarray)):
                    count += layer[k].size
                else:
                    count += 1
            else:
                count = loop_nested_layers(layer[k], count)
        return count
    return loop_nested_layers(in_dict)
