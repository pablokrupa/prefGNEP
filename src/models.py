"""
Models module for PrefGNEP surrogate models.

This module provides utilities for generating surrogate models used in
Preference-based Generalized Nash Equilibrium Problems (PrefGNEP).
Only quadratic surrogate models are currently supported.

(c) 2026 Pablo Krupa
"""

import numpy as np
import jax
import jax.numpy as jnp
from utils import gen_rand_pd_matrix, triu2array, array2triu


def gen_quad_models(sizes, full=False, diagonal=False, with_linear=False):
    """
    Generates quadratic models for each agent, to then be used as surrogate models in PrefGNEP.

    There are two options for the model structure: full or not full.
    In the full case, the model for agent i is of the form:

    .. math::
        f_i(x_i, x_{-i}) = 0.5 * x^\\top Q_i x + q_i^\\top x

    where :math:`x = (x_i, x_{-i})`. In the not full case, the model for agent i is of the form:

    .. math::
        f_i(x_i, x_{-i}) = 0.5 * x_i^\\top Q_i x_i + q_i^\\top x_i + x_{-i}^\\top A_i x_i

    the linear coupling with :math:`x_{-i}` using A_i is only introduced if `with_linear` is set to True.
    An optional argument `diagonal` can be set to True to generate diagonal Q_i matrices.

    Parameters
    ----------
    sizes : list
        List containing the number of decision variables for each agent.
    full : bool, optional
        Whether to generate full Q_i matrices or ones that only depend on x_i. Default is False.
    diagonal : bool, optional
        Whether to generate diagonal Q_i matrices or not. If True, the generated Q_i will be diagonal.
        Default is False.
    with_linear : bool, optional
        Whether to include the linear coupling term :math:`x_{-i}^T A_i x_i` or not. Only applies if `full` is False.
        Default is False.

    Returns
    -------
    th : list of dict
        List of parameter dictionaries for each agent's quadratic model.
    fs : list of callable
        List of surrogate functions for each agent.
    th_min : list of dict
        Dicts containing the lower bounds for each agent's parameters. They ensure that Q_i are positive definite.
    th_max : list of dict
        Dicts containing the upper bounds for each agent's parameters.
    """
    th = []
    fs = []
    th_min = []
    th_max = []

    N = len(sizes)  # number of agents
    dim = sum(sizes)  # total dimension

    if full:
        dim_i = [dim] * N
    else:
        dim_i = sizes

    for i in range(N):

        if diagonal:
            # Q_i = np.random.uniform(0.1, 0.1, dim_i[i])
            Q_i = np.ones(dim_i[i]) * 0.1
            lbQ = np.ones(dim_i[i]) * 0.01
            ubQ = np.inf * np.ones(dim_i[i])
        else:
            Q_i = gen_rand_pd_matrix(dim_i[i], lb_diag=0.05, ub_diag=0.1, nondiag_gain=0.0)
            # Q_i = 0.1 * np.ones((dim_i[i], dim_i[i]))
            Q_i = np.linalg.cholesky(Q_i)
            Q_i = triu2array(Q_i.T)
            lbQ = np.eye(dim_i[i]) * 0.01
            lbQ[lbQ == 0.0] = -np.inf
            lbQ = triu2array(lbQ)
            ubQ = np.inf * np.ones((dim_i[i], dim_i[i]))
            ubQ = triu2array(ubQ)
        # c_i = np.random.uniform(0.01, 0.01, dim_i[i])
        c_i = 0.01 * np.ones(sizes[i])
        th.append({'Q': jnp.asarray(Q_i), 'c': jnp.asarray(c_i)})
        th_min.append({'Q': jnp.asarray(lbQ), 'c': -jnp.inf * jnp.ones(dim_i[i])})
        th_max.append({'Q': jnp.asarray(ubQ), 'c': jnp.inf * jnp.ones(dim_i[i])})

        if not full and with_linear:
            # A_i = np.random.uniform(0.01, 0.01, (dim - sizes[i], sizes[i]))
            A_i = 0.01 * np.ones((dim - sizes[i], sizes[i]))
            lbA = -np.inf * np.ones((dim - sizes[i], sizes[i]))
            ubA = np.inf * np.ones((dim - sizes[i], sizes[i]))
            th[i]['A'] = jnp.asarray(A_i)
            th_min[i]['A'] = jnp.asarray(lbA)
            th_max[i]['A'] = jnp.asarray(ubA)

        if full:
            @jax.jit
            def f_i(th_i, x_i, x_minus_i, p=0.0):
                x_full = jnp.concatenate([x_i, x_minus_i])
                if diagonal:
                    val = 0.5 * th_i['Q'] @ (x_full**2)
                else:
                    Q_mat = array2triu(th_i['Q'], dim)
                    val = 0.5 * jnp.sum((Q_mat @ x_full)**2)
                val += th_i['c'] @ x_full
                return jnp.reshape(val, ())
        else:
            @jax.jit
            def f_i(th_i, x_i, x_minus_i, p=0.0, i=i):
                if diagonal:
                    val = 0.5 * th_i['Q'] @ (x_i**2)
                else:
                    Q_mat = array2triu(th_i['Q'], sizes[i])
                    val = 0.5 * jnp.sum((Q_mat @ x_i)**2)
                val += th_i['c'] @ x_i
                if with_linear:
                    A_i = jnp.empty((dim - sizes[i], sizes[i]), x_i.dtype)
                    A_i = A_i.at[:].set(th_i['A'])
                    val += x_minus_i @ (A_i @ x_i)
                return jnp.reshape(val, ())
        fs.append(f_i)

    return th, fs, th_min, th_max


# def gen_quad_models_old(sizes):
#     """
#     Generates quadratic models for each agent, to then be used as surrogate models in PrefGNEP.

#     Args:
#         sizes (list): List containing the number of decision variables for each agent.

#     Returns:
#         th (list): List of parameter dictionaries for each agent's quadratic model.
#         fs (list): List of surrogate functions for each agent.
#         th_min (list): dicts containing the lower bounds for each agent's parameters.
#         th_max (list): dicts containing the upper bounds for each agent's parameters.
#     """
#     th = []
#     fs = []
#     th_min = []
#     th_max = []

#     N = len(sizes)  # number of agents
#     dim = sum(sizes)  # total dimension

#     for i in range(N):

#         Q_i = gen_rand_pd_matrix(dim, lb_diag=0.05, ub_diag=0.1, nondiag_gain=0.0)  # TODO: add these as parameters
#         Q_i = np.linalg.cholesky(Q_i)
#         c_i = jnp.ones(dim) * 0.05  # TODO: add these as parameters
#         th.append({'Q': triu2array(Q_i.T), 'c': c_i})

#         lbQ = np.eye(dim) * 0.01
#         lbQ[lbQ == 0.0] = -np.inf
#         ubQ = np.inf * np.ones((dim, dim))
#         th_min.append({'Q': triu2array(lbQ), 'c': -np.inf * jnp.ones(dim)})
#         th_max.append({'Q': triu2array(ubQ), 'c': np.inf * jnp.ones(dim)})

#         @jax.jit
#         def f_i(th_i, x_i, x_minus_i, p=0.0):
#             x_full = jnp.concatenate([x_i, x_minus_i])
#             Q_mat = array2triu(th_i['Q'], dim)
#             val = 0.5 * jnp.sum((Q_mat @ x_full)**2) + th_i['c'] @ x_full
#             return jnp.reshape(val, ())

#         fs.append(f_i)

#     return th, fs, th_min, th_max


# def gen_small_quad_models(sizes, with_linear: bool = False):
#     """
#     Generates a quadratic model for x_i with linear coupling with x_-i, to be used as surrogate models in PrefGNEP.

#     The model for agent i is of the form:
#     f_i(x_i, x_-i) = 0.5 * x_i^T Q_i x_i + c_i^T x_i + x_-i^T A_i x_i

#     Args:
#         sizes (list): List containing the number of decision variables for each agent.
#         with_linear (bool): Whether to include the linear coupling term x_-i^T A_i x_i in the model or not.

#     Returns:
#         th (list): List of parameter dictionaries for each agent's quadratic model.
#         fs (list): List of surrogate functions for each agent.
#         th_min (list): dicts containing the lower bounds for each agent's parameters.
#         th_max (list): dicts containing the upper bounds for each agent's parameters.
#     """
#     th = []
#     fs = []
#     th_min = []
#     th_max = []

#     N = len(sizes)  # number of agents
#     dim = sum(sizes)  # total dimension

#     for i in range(N):

#         Q_i = gen_rand_pd_matrix(sizes[i], lb_diag=0.05, ub_diag=0.1, nondiag_gain=0.0)  # TODO: add these as parameters
#         Q_i = np.linalg.cholesky(Q_i)
#         c_i = jnp.ones(sizes[i]) * 0.05  # TODO: add these as parameters
#         if with_linear:
#             A_i = jnp.ones((dim - sizes[i], sizes[i])) * 0.05  # TODO: add these as parameters
#             th.append({'Q': triu2array(Q_i.T), 'c': c_i, 'A': A_i})
#         else:
#             th.append({'Q': triu2array(Q_i.T), 'c': c_i})

#         lbQ = np.eye(sizes[i]) * 0.01
#         lbQ[lbQ == 0.0] = -np.inf
#         ubQ = np.inf * np.ones((sizes[i], sizes[i]))
#         if with_linear:
#             th_min.append({'Q': triu2array(lbQ), 'c': -np.inf * jnp.ones(sizes[i]),
#                            'A': -np.inf * jnp.ones((dim - sizes[i], sizes[i]))})
#             th_max.append({'Q': triu2array(ubQ), 'c': np.inf * jnp.ones(sizes[i]),
#                            'A': np.inf * jnp.ones((dim - sizes[i], sizes[i]))})
#         else:
#             th_min.append({'Q': triu2array(lbQ), 'c': -np.inf * jnp.ones(sizes[i])})
#             th_max.append({'Q': triu2array(ubQ), 'c': np.inf * jnp.ones(sizes[i])})

#         @jax.jit
#         def f_i(th_i, x_i, x_minus_i, p=0.0, i=i):
#             Q_mat = array2triu(th_i['Q'], sizes[i])
#             val = 0.5 * jnp.sum((Q_mat @ x_i)**2) + th_i['c'] @ x_i
#             if with_linear:
#                 A_i = jnp.empty((dim - sizes[i], sizes[i]), x_i.dtype)
#                 A_i = A_i.at[:].set(th_i['A'])
#                 val += x_minus_i @ (A_i @ x_i)
#             return jnp.reshape(val, ())

#         fs.append(f_i)

#     return th, fs, th_min, th_max


# def gen_diag_quad_models(sizes, with_linear: bool = False):
#     """
#     Generates a quadratic model for x_i, with diagonal Q_i, to be used as surrogate models in PrefGNEP.

#     The model for agent i is of the form:
#     f_i(x_i, x_-i) = 0.5 * x_i^T Q_i x_i + c_i^T x_i

#     Args:
#         sizes (list): List containing the number of decision variables for each agent.
#     Returns:
#         th (list): List of parameter dictionaries for each agent's quadratic model.
#         fs (list): List of surrogate functions for each agent.
#         th_min (list): dicts containing the lower bounds for each agent's parameters.
#         th_max (list): dicts containing the upper bounds for each agent's parameters.
#     """
#     # TODO: Merge this with gen_quad_models by adding an argument to choose between diagonal and full Q_i.

#     th = []
#     fs = []
#     th_min = []
#     th_max = []

#     N = len(sizes)  # number of agents
#     dim = sum(sizes)  # total dimension

#     for i in range(N):
#         Q_i = jnp.ones(sizes[i]) * 0.1
#         c_i = jnp.ones(sizes[i]) * 0.01
#         if with_linear:
#             A_i = jnp.ones((dim - sizes[i], sizes[i])) * 0.01  # TODO: add these as parameters
#             th.append({'Q': Q_i, 'c': c_i, 'A': A_i})
#         else:
#             th.append({'Q': Q_i, 'c': c_i})

#         if with_linear:
#             th_min.append({'Q': 0.01 * jnp.ones(sizes[i]), 'c': -np.inf * jnp.ones(sizes[i]),
#                            'A': -np.inf * jnp.ones((dim - sizes[i], sizes[i]))})
#             th_max.append({'Q': np.inf * jnp.ones(sizes[i]), 'c': np.inf * jnp.ones(sizes[i]),
#                            'A': np.inf * jnp.ones((dim - sizes[i], sizes[i]))})
#         else:
#             th_min.append({'Q': 0.01 * jnp.ones(sizes[i]), 'c': -np.inf * jnp.ones(sizes[i])})
#             th_max.append({'Q': np.inf * jnp.ones(sizes[i]), 'c': np.inf * jnp.ones(sizes[i])})

#         @jax.jit
#         def f_i(th_i, x_i, x_minus_i, p=0.0, i=i):
#             val = 0.5 * th_i['Q'] @ (x_i**2) + th_i['c'] @ x_i
#             if with_linear:
#                 A_i = jnp.empty((dim - sizes[i], sizes[i]), x_i.dtype)
#                 A_i = A_i.at[:].set(th_i['A'])
#                 val += x_minus_i @ (A_i @ x_i)
#             return jnp.reshape(val, ())

#         fs.append(f_i)

#     return th, fs, th_min, th_max
