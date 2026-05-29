"""
Centralised definitions of all GNEP benchmark problems used by the prefGNEP examples.

Each problem is returned as a GNEPdata dataclass instance.  Retrieve a problem by
name with get_GNEP(name).

Available problems
------------------
  "Picheny"      – Picheny, Binois & Habbal (2019), Section 4.1
  "Facchinei_A3" – Facchinei & Kanzow (2009), Example A.3
  "Pavel_Ex1"    – Salehisadaghiani, Shi & Pavel (2017), Example 1

(c) 2026 Pablo Krupa
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GNEPdata:
    """Container for a GNEP problem definition.

    Attributes
    ----------
    name : str
        Short identifier used for file names, registry lookups, etc.
    sizes : list of int
        Number of decision variables per agent.
    f_real : list of callable
        True objective functions, one per agent.  Each callable must accept
        ``(x, p=0.0)`` and return a scalar.
    lb : array-like
        Lower bounds for the joint decision vector.
    ub : array-like
        Upper bounds for the joint decision vector.
    Aeq : array-like or None
        Matrix for linear equality constraints  Aeq @ x == beq.
    beq : array-like or None
        Right-hand side of linear equality constraints.
    g : callable or None
        Shared inequality constraint function  g(x, p=0.0) <= 0.
    ng : int or None
        Number of inequality constraints (rows of g).
    h : callable or None
        Shared equality constraint function  h(x, p=0.0) == 0.
    nh : int or None
        Number of equality constraints (rows of h).
    quad_model_kwargs : dict
        Keyword arguments forwarded to ``gen_quad_models``.
    dataset_file : str
        Default file name (no path) used to save/load the initial dataset.
    dist_tol : float
        Tolerance for the distance-to-GNE metric (used for reference lines in plots).
    br_tol : float
        Tolerance for the BR-deviation metric (used for reference lines in plots).
    """
    name: str
    sizes: List[int]
    f_real: List[Any]
    lb: Any
    ub: Any
    quad_model_kwargs: Dict[str, Any]
    dataset_file: str
    Aeq: Optional[Any] = None
    beq: Optional[Any] = None
    g: Optional[Any] = None
    ng: Optional[int] = None
    h: Optional[Any] = None
    nh: Optional[int] = None
    dist_tol: Optional[float] = 0.02
    br_tol: Optional[float] = 0.02


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

def get_picheny() -> GNEPdata:
    """Return the Picheny (2019) Section 4.1 problem.

    2 agents, 1 decision variable each, no shared constraints.
    """
    @jax.jit
    def f1(x, p=0.0):
        val = (x[1] - 5.1 * (x[0] / (2.0 * jnp.pi))**2 + 5 * x[0] / jnp.pi - 6.0)**2
        val += 10.0 * (1.0 - 1.0 / (8.0 * jnp.pi)) * jnp.cos(x[0]) + 10.0
        return val

    @jax.jit
    def f2(x, p=0.0):
        val = -jnp.sqrt((10.5 - x[0]) * (x[0] + 5.5) * (x[1] + 0.5))
        val -= (x[1] - 5.1 * (x[0] / (2.0 * jnp.pi))**2 - 6.0)**2 / 30.0
        val -= ((1.0 - 1.0 / (8.0 * jnp.pi)) * jnp.cos(x[0]) + 1.0) / 3.0
        return val

    return GNEPdata(
        name="Picheny",
        sizes=[1, 1],
        f_real=[f1, f2],
        lb=np.array([-5.0, 0.0]),
        ub=np.array([10.0, 15.0]),
        quad_model_kwargs={"full": False, "with_linear": False},
        dataset_file="example_picheny_dataset.pkl",
        dist_tol=0.005331,
        br_tol=0.005331,
    )


def get_facchinei_A3() -> GNEPdata:
    """Return the Facchinei & Kanzow (2009) Example A.3 problem.

    3 agents (sizes=[3,2,2]), quadratic objectives, 4 shared inequality
    constraints.
    """
    sizes = [3, 2, 2]
    N = len(sizes)

    A = [jnp.array([[20, 5, 3], [5, 5, -5], [3, -5, 15]]),
         jnp.array([[11, -1], [-1, 9]]),
         jnp.array([[48, 39], [39, 53]])]
    B = [jnp.array([[-6, 10, 11, 20], [10, -4, -17, 9], [15, 8, -22, 21]]),
         jnp.array([[20, 1, -3, 12, 1], [10, -4, 8, 16, 21]]),
         jnp.array([[10, -2, 22, 12, 16], [9, 19, 21, -4, 20]])]
    b = [jnp.array([1, -1, 1]), jnp.array([1, 0]), jnp.array([-1, 2])]

    idx_i = []
    idx_minus_i = []
    current_index = 0
    for i, size in enumerate(sizes):
        idx_i_list = list(range(current_index, current_index + size))
        minus_i_indices = (list(range(0, current_index))
                           + list(range(current_index + size, sum(sizes))))
        idx_i.append(jnp.array(idx_i_list))
        idx_minus_i.append(jnp.array(minus_i_indices))
        current_index += size

    f_real = []
    for i in range(N):
        @jax.jit
        def fi(x, p=0.0, i=i):
            val = (0.5 * x[idx_i[i]].T @ A[i] @ x[idx_i[i]]
                   + x[idx_i[i]].T @ (B[i] @ x[idx_minus_i[i]] + b[i]))
            return jnp.reshape(val, ())
        f_real.append(fi)

    lb = -10.0 * jnp.ones(sum(sizes))
    ub = 10.0 * jnp.ones(sum(sizes))

    A_ineq = jnp.array([[1, 1, 1, 0, 0, 0, 0],
                        [1, 1, -1, -1, 0, 0, 1],
                        [0, -1, -1, 1, -1, 1, 0],
                        [-1, 0, -1, 1, 0, 0, 1]])
    b_ineq = jnp.array([20.0, 5.0, 7.0, 4.0])
    ng = int(A_ineq.shape[0])

    def g(x, p=0.0):
        return A_ineq @ x - b_ineq

    return GNEPdata(
        name="Facchinei_A3",
        sizes=sizes,
        f_real=f_real,
        lb=lb,
        ub=ub,
        g=g,
        ng=ng,
        quad_model_kwargs={"full": False, "diagonal": False, "with_linear": True},
        dataset_file="example_Facchinei_A3_dataset.pkl",
        dist_tol=0.061335,
        br_tol=0.065385,
    )


def get_pavel_ex1() -> GNEPdata:
    """Return the Pavel / Fabiani–Bemporad (2024) Section VI.B Example 1 problem.

    10 agents, 1 decision variable each, no shared constraints.
    """
    N = 10
    sizes = [1] * N
    dim = sum(sizes)

    f_real = []
    for i in range(N):
        @jax.jit
        def fi(x, p=0.0, i=i):
            val = N * (1.0 + i / 2.0) * x[i] - x[i] * (60.0 * N - jnp.sum(x))
            return val
        f_real.append(fi)

    lb = 7.0 * np.ones(dim)
    ub = 100.0 * np.ones(dim)

    return GNEPdata(
        name="Pavel_Ex1",
        sizes=sizes,
        f_real=f_real,
        lb=lb,
        ub=ub,
        quad_model_kwargs={"full": False, "with_linear": True},
        dataset_file="example_Pavel_Ex1_dataset.pkl",
        dist_tol=0.003422,
        br_tol=0.001198,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_REGISTRY = {
    "Picheny": get_picheny,
    "Facchinei_A3": get_facchinei_A3,
    "Pavel_Ex1": get_pavel_ex1,
}


def get_GNEP(name: str) -> GNEPdata:
    """Return a GNEPdata instance for the requested problem.

    Parameters
    ----------
    name : str
        Problem identifier.  One of: ``"Picheny"``, ``"Facchinei_A3"``, ``"Pavel_Ex1"``.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise ValueError(
            f"Unknown problem '{name}'. Available problems: {available}."
        )
    return _REGISTRY[name]()
