"""
Module implementing preference-based learning for Generalized Nash Equilibrium Problems (GNEPs).
Provides tools for fitting surrogate cost functions from pairwise preference data and solving GNEPs using active learning.

(c) 2026 Pablo Krupa
"""

import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP, ParametricGNEP
import jaxopt
import copy
from tqdm import tqdm
import pyswarms as ps
from jax.typing import ArrayLike
from typing import Callable, NamedTuple
from types import SimpleNamespace
from timeit import default_timer as timer
from dataset import DataSet, dist_registry

jax.config.update("jax_enable_x64", True)


class MinimizeInfo(NamedTuple):
    """Class for storing information about solving the fitting problem, see PrefGNEP._fit() method."""
    lbfgs_solved: bool  # Indicates if the L-BFGS solver termineted successfully
    lbfgs_iters: int  # Number of L-BFGS iterations
    adam_iters: int  # Number of Adam iterations performed
    time_setup: float  # Time [s] spent setting up the fitting problem
    time_adam: float  # Time [s] spent by the Adam solver
    time_lbfgs: float  # Time [s] spent by the L-BFGS solver
    time_total: float  # Total time [s] required to solve the problem. Includes overhead time of Classifier.fit()
    lbfgs_fun_eval: int  # Number of function evaluations of the L-BFGS solver
    lbfgs_jac_eval: int  # Number of Jacobian evaluations of the L-BFGS solver
    lbfgs_hess_eval: int  # Number of Hessian evaluations of the L-BFGS solver


class ALloopInfo(NamedTuple):
    """
    Class to store the information of the active learning loop.

    For additional information or context, see the PrefGNEP.fit_AL_loop() method.

    Attributes
    ----------
    x_star : list of ArrayLike
        List of learned GNE solutions at each iteration.
    accuracy : list of float
        List of accuracy scores at each iteration, in terms of prediction of preferences in dataset.
    times : list of float
        List of cumulative times [s] at each iteration.
    eval : list of float
        List of evaluation metrics at each iteration, measured by the f_eval argument passed to fit_AL_loop().
    tol_gnep : list of float
        List of relative GNEP solution changes used for the stopping criterion at each iteration.
    tol_eval : list of float
        List of evaluation function norms used for the stopping criterion at each iteration.
    res_x_star : list of float
        List of KKT residuals (inf-norm) at the learned GNE solution at each iteration.
    res_ds_gnep : list of float
        List of GNEP KKT residuals (inf-norm) when augmenting the dataset at each iteration.
    res_ds_br : list of list of bool
        List of best response solver success flags for each player when augmenting the dataset at each iteration.
    delta : list of float
        List of exploration parameter values at each iteration.
    sigma : list of float
        List of noise parameter values at each iteration.
    mu_th : list of float
        List of regularization parameter mu_th values at each iteration.
    n_iters : int
        Total number of iterations performed in the AL loop.
    best_th : dict or list of ArrayLike or None
        Best surrogate model parameters found during the AL loop, selected according to the
        minimum value of f_eval at the GNEP solution. Only stored if store_best_th is True
        and f_eval is provided in fit_AL_loop().
    """
    x_star: list[ArrayLike] = []
    accuracy: list[float] = []
    times: list[float] = []
    eval: list[float] = []
    tol_gnep: list[float] = []
    tol_eval: list[float] = []
    res_x_star: list[float] = []
    res_ds_gnep: list[float] = []
    res_ds_br: list[list[bool]] = []
    delta: list[float] = []
    sigma: list[float] = []
    mu_th: list[float] = []
    n_iters: int = 0
    best_th: dict | list[ArrayLike] | None = None

    def to_array(self):
        """Converts the lists in ALLoopInfo to numpy arrays."""
        d = self._asdict()
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = np.array(v)
        return ALloopInfo(**d)

    def to_list(self):
        """Converts the numpy arrays in ALLoopInfo to lists."""
        d = self._asdict()
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return ALloopInfo(**d)

    def assign_n_iters(self):
        """Sets the n_iters field to the length of the times list."""
        return self._replace(n_iters=len(self.times))


def adam_solver(JdJ, z, solver_iters, adam_eta, verbose, params_min=None, params_max=None):
    """
    Solves a nonlinear optimization problem using the Adam optimization algorithm.

    Parameters
    ----------
    JdJ : callable
        A function that computes the objective function value and its gradient.
    z : dict or list
        A dict or list of numpy arrays representing the initial guess of the optimization variables.
    solver_iters : int
        The number of iterations for the solver.
    adam_eta : float
        The learning rate for the Adam algorithm.
    verbose : int
        Verbosity level. Set to 0 to disable printing.
    params_min : dict or list, optional
        A dict or list of numpy arrays representing the lower bounds of the optimization variables z.
        Lower bounds are enforced by clipping the variables during the iterations.
    params_max : dict or list, optional
        A dict or list of numpy arrays representing the upper bounds of the optimization variables z.
        Upper bounds are enforced by clipping the variables during the iterations.

    Returns
    -------
    tuple
        A tuple containing the optimized variables `z` and the objective function value `Jopt`.

    Note: Function taken from https://github.com/bemporad/jax-sysid
    """
    # Flatten z
    z_l, z_def = jax.tree_util.tree_flatten(z)

    if verbose > 0:
        iters_tqdm = tqdm(total=solver_iters, desc='Iterations', ncols=30,
                          bar_format='{percentage:3.0f}%|{bar}|', leave=True, position=0)
        loss_log = tqdm(total=0, position=1, bar_format='{desc}')
        nvars = sum([zi.size for zi in z_l])
        print("Solving NLP with Adam (%d optimization variables) ..." % nvars)

    nz = len(z_l)
    fbest = np.inf
    v = [np.zeros(zi.shape) if hasattr(zi, 'shape') else 0.0 for zi in z_l]
    m = [np.zeros(zi.shape) if hasattr(zi, 'shape') else 0.0 for zi in z_l]
    beta1 = 0.9
    beta2 = 0.999
    beta1t = beta1
    beta2t = beta2
    epsil = 1e-8

    ismin = (params_min is not None)
    ismax = (params_max is not None)
    if params_min is not None:
        params_min_l, _ = jax.tree_util.tree_flatten(params_min)
    if params_max is not None:
        params_max_l, _ = jax.tree_util.tree_flatten(params_max)
    isbounded = ismin or ismax

    if isbounded:
        # Clip the initial guess, when required
        for j in range(nz):
            if ismin:
                z_l[j] = np.maximum(z_l[j], params_min_l[j])
            if ismax:
                z_l[j] = np.minimum(z_l[j], params_max_l[j])

    for k in range(solver_iters):
        f, df = JdJ(z_l, z_def)
        df_l, _ = jax.tree_util.tree_flatten(df)
        if fbest > f:
            fbest = f
            zbest = z_l.copy()
        for j in range(nz):
            m[j] = beta1 * m[j] + (1 - beta1) * df_l[j]
            v[j] = beta2 * v[j] + (1 - beta2) * df_l[j]**2
            # classical Adam step
            z_l[j] -= adam_eta / (1 - beta1t) * m[j] / (np.sqrt(v[j] / (1 - beta2t)) + epsil)
        if isbounded:
            for j in range(nz):
                if ismin:
                    z_l[j] = np.maximum(z_l[j], params_min_l[j])
                if ismax:
                    z_l[j] = np.minimum(z_l[j], params_max_l[j])

        beta1t *= beta1
        beta2t *= beta2

        if verbose > 0:
            str = f"    f = {f: 10.6f}, f* = {fbest: 8.6f}"
            ndf = np.sum([np.linalg.norm(df_l[i])
                          for i in range(nz)])
            str += f", |grad f| = {ndf: 8.6f}"
            str += f", iter = {k + 1}"
            loss_log.set_description_str(str)
            iters_tqdm.update(1)

    z_l = zbest.copy()
    Jopt, _ = JdJ(z_l, z_def)  # = fbest
    z_sol = z_def.unflatten(z_l)

    if verbose > 0:
        iters_tqdm.close()
        loss_log.close()

    return z_sol, Jopt


class PrefGNEP:
    """
     A class for solving Generalized Nash Equilibrium Problems (GNEPs) using preference-based learning.

    Learns surrogate cost functions for each player from pairwise preference data, and uses them to find
    approximate Generalized Nash Equilibria (GNE). Learning is done using an Active Learning (AL) loop,
    where the dataset of preferences is iteratively augmented with new samples.
    See the fit_AL_loop() method for more details on the AL loop and its parameters.
    See also solve_gnep() and best_response() methods for solving the GNEP and best response problems,
    respectively, for the current surrogate cost functions.
    """

    def __init__(self, sizes: list[int], fc: list[callable], g: list[callable] = None, ng: int = None,
                 h: list[callable] = None, nh: int = None, lb: ArrayLike = None, ub: ArrayLike = None,
                 Aeq: ArrayLike = None, beq: ArrayLike = None, npar: int = 0):
        """
        Initialize the PrefGNEP object.

        Parameters
        ----------
        sizes : list[int]
            List of sizes of each player's decision variables.
        fc : list[callable]
            List of callables for each player's surrogate cost function.
            Each callable should have the signature: fc_i(th, x_i, x_-i),
            where th are the parameters, x_i are the player's decision variables,
            and x_-i are the other players' decision variables.
        g : list[callable], optional
            List of inequality constraint functions for each player. Default is None.
        ng : int, optional
            Number of inequality constraints per player. Default is None.
        h : list[callable], optional
            List of equality constraint functions for each player. Default is None.
        nh : int, optional
            Number of equality constraints per player. Default is None.
        lb : ArrayLike, optional
            Lower bounds for each player's decision variables. Default is None.
        ub : ArrayLike, optional
            Upper bounds for each player's decision variables. Default is None.
        Aeq : ArrayLike, optional
            Equality constraint matrix for shared constraints. Default is None.
        beq : ArrayLike, optional
            Equality constraint vector for shared constraints. Default is None.
        npar : int, optional
            Number of parameters when dealing with a parametric GNEP.
            If 0, the problem is treated as a non-parametric GNEP. Default is 0.

        Notes
        -----
        The number of players `N` is inferred from the length of `sizes`, and
        the total dimension of decision variables `dim` is computed as the sum
        of all elements in `sizes`.

        See nashopt.GNEP and nashopt.ParametricGNEP for more details on the problem formulation and
        the meaning of the parameters, as well as the expression of the GNEP problem.
        """
        self.sizes = sizes  # List of sizes of each player's decision variables
        self.N = len(sizes)  # Number of players
        self.dim = sum(sizes)  # Total dimension of decision variables
        self.fc = fc  # List of callables for each player's surrogate cost function. Signature: fc_i(th, x_i, x_-i)
        self.g = g  # List of constraint functions for each player
        self.ng = ng  # Number of constraints per player
        self.h = h  # List of equality constraint functions for each player
        self.nh = nh  # Number of equality constraints per player
        self.lb = lb  # Lower bounds for each player's decision variables
        self.ub = ub  # Upper bounds for each player's decision variables
        self.Aeq = Aeq  # Equality constraint matrix
        self.beq = beq  # Equality constraint vector
        self.npar = npar  # Number of parameters when dealing with parametric GNEP. If 0, we have a non-parametric GNEP.
        self._parametric = True if npar > 0 else False
        self._opt_ps = {'c1': 2.8, 'c2': 1.3, 'w': 0.9}  # Options for the particle swarm optimizer
        self._compute_idx()

    def _compute_idx(self):
        """
        Compute the start and end indices for each player's decision variables in the concatenated decision vector.
        Also stores the indices of the -i variables for each player.
        """
        self.idx_i = []
        self.idx_minus_i = []
        current_index = 0
        for size in self.sizes:
            idx_i_list = list(range(current_index, current_index + size))
            self.idx_i.append(idx_i_list)
            minus_i_indices = list(range(0, current_index)) + list(range(current_index + size, sum(self.sizes)))
            self.idx_minus_i.append(minus_i_indices)
            current_index += size

    def init(self, th: list[list[ArrayLike] | dict],
             th_min: list[list[ArrayLike] | dict] = None, th_max: list[list[ArrayLike] | dict] = None,
             alpha1: float = 0.0, alpha2: float = 0.0, pmin: ArrayLike = None, pmax: ArrayLike = None) -> None:
        """
        Initialize the PrefGNEP with given parameters.

        Parameters
        ----------
        th : list of list[ArrayLike] or dict
            Parameters for each player's surrogate model.
        th_min : list of list[ArrayLike] or dict, optional
            Minimum bounds for the parameters of each player's surrogate model.
        th_max : list of list[ArrayLike] or dict, optional
            Maximum bounds for the parameters of each player's surrogate model.
        alpha1 : float, optional
            1-norm regularization parameter for state in parametric GNEP. Used when self._parametric is True.
        alpha2 : float, optional
            2-norm regularization parameter for state in parametric GNEP. Used when self._parametric is True.
        pmin : ArrayLike, optional
            Minimum bounds for the parameters in parametric GNEP. Used when self._parametric is True.
        pmax : ArrayLike, optional
            Maximum bounds for the parameters in parametric GNEP. Used when self._parametric is True.
        """
        self.th = th
        self.th_min = th_min if th_min is not None else [None for _ in range(self.N)]
        self.th_max = th_max if th_max is not None else [None for _ in range(self.N)]
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.pmin = pmin
        self.pmax = pmax

    def loss(self, rho_th: float = 0.001, mu_th: float = 0.0, epsilon: float = 1.e-6) -> None:
        """
        Initialize the parameters for the loss function, see PrefGNEP.fit() method.

        Parameters
        ----------
        rho_th : float, optional
            Regularization parameter for the loss function. Must be non-negative.
        mu_th : float, optional
            Regularization parameter for the difference from initial parameters. Must be non-negative.
        epsilon : float, optional
            Small constant to avoid numerical issues in logarithms. Must be non-negative.
        """
        assert rho_th >= 0.0, "Expected rho_th to be non-negative"
        assert mu_th >= 0.0, "Expected mu_th to be non-negative"
        assert epsilon >= 0.0, "Expected epsilon to be non-negative"
        self.rho_th = rho_th
        self.mu_th = mu_th
        self.epsilon = epsilon

    def optimization(self, adam_eta: float = 0.001, adam_epochs: int = 0, lbfgs_epochs: int = 1000, verbose: int = -1,
                     memory: int = 10, lbfgs_tol: float = 1.e-16, lbfgs_maxiter: int = 1000, lbfgs_maxls: int = 20,
                     jit: bool = True) -> None:
        """
        Initialize the parameters for the optimization of the learning problem, see PrefGNEP.fit() method.

        Parameters
        ----------
        adam_eta : float, optional
            Learning rate for the Adam optimizer.
        adam_epochs : int, optional
            Number of epochs for the Adam optimizer (0 means no Adam optimization, and just L-BFGS-B).
        lbfgs_epochs : int, optional
            Maximum number of function evaluations for the L-BFGS-B optimizer.
        verbose : int, optional
            Verbosity level for the optimizers.
        memory : int, optional
            Memory parameter for the L-BFGS-B optimizer.
        lbfgs_tol : float, optional
            Tolerance for the L-BFGS-B optimizer.
        lbfgs_maxiter : int, optional
            Maximum number of iterations for the L-BFGS-B optimizer.
        lbfgs_maxls : int, optional
            Maximum number of line search steps for the L-BFGS-B optimizer.
        jit : bool, optional
            Whether to JIT compile the optimization functions.
        """
        assert adam_eta >= 0.0, "Expected adam_eta to be non-negative"
        assert adam_epochs >= 0, "Expected adam_epochs to be non-negative"
        assert lbfgs_epochs >= 0, "Expected lbfgs_epochs to be non-negative"
        assert memory > 0, "Expected memory to be positive"
        assert lbfgs_tol >= 0.0, "Expected lbfgs_tol to be non-negative"
        assert lbfgs_maxiter > 0, "Expected lbfgs_maxiter to be positive"
        assert lbfgs_maxls > 0, "Expected lbfgs_maxls to be positive"
        self.adam_eta = adam_eta
        self.adam_epochs = adam_epochs
        self.lbfgs_epochs = lbfgs_epochs
        self.verbose = verbose
        self.memory = memory
        self.lbfgs_tol = lbfgs_tol
        self.lbfgs_maxiter = lbfgs_maxiter
        self.lbfgs_maxls = lbfgs_maxls
        self.jit = jit

    def fit(self, ds: DataSet, mu_th: float = None, update: bool = True,
            th_0: list[list[ArrayLike] | dict] = None) -> tuple[list[ArrayLike | dict], list[MinimizeInfo], SimpleNamespace]:
        """
        Fit the surrogate models using the provided preference data in the dataset `ds`.
        See PrefGNEP.loss() method for setting the parameters of the loss function.
        See PrefGNEP.optimization() method for setting the parameters of the optimization process.
        See PrefGNEP._fit_i() method for more details on the fitting process for each player.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.
        update : bool, optional
            Whether to update self.th with the learned parameters after fitting. Default is True.
        th_0 : list of list[ArrayLike] or dict, optional
            Initial value of the parameters for the laerning problem solver. If None, the current value
            of self.th is used as the initial condition for the optimization.
        mu_th : float, optional
            Regularization parameter for the difference from initial parameters. Must be non-negative.

        Returns
        -------
        th : list of ArrayLike or dict
            Learned parameters for each player's surrogate model.
        infos : list of MinimizeInfo
            Optimization information for each player, including convergence details.
        stats : SimpleNamespace
            Statistics of the fitting process, including:
            - time : float
                Total time taken to fit all surrogate models.
        """
        if mu_th is not None:
            assert mu_th >= 0.0, "Expected mu_th to be non-negative"
        infos = []
        if th_0 is None:
            th_0 = [None for _ in range(self.N)]
        for i in range(self.N):
            _, info_lbfgs_i = self._fit_i(i, ds.samples, ds.xA[i], ds.xB[i], ds.prefs[i], dist_i=ds.dist[i],
                                          mu_th=mu_th, update=update, th_0=th_0[i])
            infos.append(info_lbfgs_i)

        stats = SimpleNamespace(time=0.0)
        for info in infos:
            stats.time += info.time_total
        return self.th, infos, stats

    def _fit_i(self, i: int, samples: ArrayLike, xA_i: ArrayLike, xB_i: ArrayLike, prefs_i: ArrayLike,
               dist_i: ArrayLike = None, mu_th: float = None, update: bool = True,
               th_0: list[ArrayLike] | dict = None) -> tuple[ArrayLike | dict, MinimizeInfo]:
        """
        Fit the surrogate model for player i using the provided preference data. The method returns the
        learned parameters for player i's surrogate model and optimization information about the fitting process.
        It also updates the parameters of self.th[i] with the learned parameters if `update` is True (default behavior).
        See DataSet class for more details on the expected format of the input data.

        Parameters
        ----------
        i : int
            Index of the player.
        samples : ArrayLike
            Dataset of decision variables.
        xA_i : ArrayLike
            Dataset of preference pairs A for agent i.
        xB_i : ArrayLike
            Dataset of preference pairs B for agent i.
        prefs_i : ArrayLike
            Preference labels for agent i.
        dist_i : ArrayLike, optional
            Distance-based weights for each sample in the dataset.
        th_0 : list[ArrayLike] or dict, optional
            Initial value of the parameters for the learning problem solver. If None, the current value
            of self.th[i] is used as the initial condition for the optimization.
        mu_th : float, optional
            Regularization parameter for the difference from initial parameters.
        update : bool, optional
            Whether to update self.th[i] with the learned parameters. Default is True.

        Returns
        -------
        th_i: ArrayLike or dict
            Learned parameters for player i's surrogate model.
        info: MinimizeInfo
            Optimization information for the fitting process of player i, including convergence details.
        """
        start_fit = timer()
        th_i = copy.deepcopy(th_0) if th_0 is not None else copy.deepcopy(self.th[i])
        th_curr = copy.deepcopy(self.th[i])
        _mu_th = mu_th if mu_th is not None else self.mu_th

        if dist_i is None:
            dist_i = jnp.ones_like(prefs_i)  # If no distance weights are provided, use uniform weights (i.e., no weighting)

        @jax.jit
        def loss_fnc(th: ArrayLike) -> float:
            """
            Computes the loss for player i given parameters th.
            """
            M = samples.shape[0]  # Size of dataset

            @jax.jit
            def single_loss(x_j: ArrayLike, xA_j: ArrayLike, xB_j: ArrayLike, p_j: float, dist_j: float) -> float:
                fA = self.fc[i](th, xA_j, self.x_mi(x_j, i))
                fB = self.fc[i](th, xB_j, self.x_mi(x_j, i))
                P = 1. / (1. + jnp.exp((fA - fB) / (dist_j)))
                V = - p_j * jnp.log(P + self.epsilon) - (1. - p_j) * jnp.log(1. - P + self.epsilon)
                return V

            total_loss = jnp.sum(jax.vmap(single_loss, in_axes=(0, 0, 0, 0, 0))(samples, xA_i, xB_i, prefs_i, dist_i)) / M

            for param, param_0 in zip(jax.tree_util.tree_leaves(th), jax.tree_util.tree_leaves(th_curr)):
                total_loss += self.rho_th * (jnp.sum(param ** 2))
                total_loss += _mu_th * jnp.sum((param - param_0) ** 2)
            return total_loss

        # Run Adam solver
        start_adam = timer()

        def JdJ(z_leaves, z_treedef):
            return jax.value_and_grad(loss_fnc)(z_treedef.unflatten(z_leaves))

        if self.adam_epochs > 0:
            th_i, loss_adam = adam_solver(JdJ, th_i, self.adam_epochs, self.adam_eta,
                                          self.verbose, self.th_min[i], self.th_max[i])
        end_adam = timer()

        # Run L-BFGS solver
        start_lbfgs = timer()
        # TODO: Use ScipyMinimize if we have no bounds
        solver = jaxopt.ScipyBoundedMinimize(fun=loss_fnc, method='L-BFGS-B',
                                             jit=self.jit,
                                             tol=self.lbfgs_tol,
                                             maxiter=self.lbfgs_maxiter,
                                             options={"maxls": self.lbfgs_maxls, 'iprint': self.verbose,
                                                      'gtol': self.lbfgs_tol, 'ftol': self.lbfgs_tol,
                                                      'maxcor': self.memory, 'maxfun': self.lbfgs_epochs})

        bounds = (self.th_min[i], self.th_max[i]) if (self.th_min[i] is not None and self.th_max[i] is not None) else None
        th_i, info_lbfgs = solver.run(th_i, bounds=bounds)
        end_lbfgs = timer()

        if update:
            self.th[i] = th_i

        # Collect and return info
        if self.verbose > 0:
            if info_lbfgs.success:
                print("L-BFGS-B solver was successful!")
            else:
                print("Error solving fitting problem using L-BFGS-B solver")
        info = MinimizeInfo(lbfgs_solved=info_lbfgs.success,
                            lbfgs_iters=info_lbfgs.iter_num,
                            adam_iters=self.adam_epochs,
                            time_setup=start_adam - start_fit,
                            time_adam=end_adam - start_adam,
                            time_lbfgs=end_lbfgs - start_lbfgs,
                            time_total=timer() - start_fit,
                            lbfgs_fun_eval=int(info_lbfgs.num_fun_eval),
                            lbfgs_jac_eval=int(info_lbfgs.num_jac_eval),
                            lbfgs_hess_eval=int(info_lbfgs.num_hess_eval))
        return th_i, info

    def predict_dataset(self, ds: DataSet) -> ArrayLike:
        """
        Predict preferences for given dataset using the learned surrogate models.

        Preferences are predicted using the surrogate functions self.fc and the learned parameters
        self.th. A preference of 1.0 indicates that option A is preferred, while 0.0 indicates that
        option B is preferred.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples and preference pairs.

        Returns
        -------
        list of numpy.ndarray
            Predicted preferences for all agents.
        """
        M = ds.size  # Size of dataset
        prefs_pred = [[] for _ in range(self.N)]

        for i in range(self.N):
            def single_predict(x_j: ArrayLike, xA_j: ArrayLike, xB_j: ArrayLike, dist_j: float) -> float:
                prob = self.predict_proba_i(i, xA_j, xB_j, x_j, dist_j)
                return 1.0 if prob >= 0.5 else 0.0

            for j in range(M):
                prefs_pred[i].append(single_predict(ds.samples[j], ds.xA[i][j], ds.xB[i][j], ds.dist[i][j]))
        prefs_pred = [np.array(arr) for arr in prefs_pred]

        return prefs_pred

    def predict_pair(self, x1: ArrayLike, x2: ArrayLike) -> list[float]:
        """
        Predict preferences between two decision variable vectors for all agents.

        Parameters
        ----------
        x1 : ArrayLike
            Decision variable vector for option A.
        x2 : ArrayLike
            Decision variable vector for option B.

        Returns
        -------
        list of float
            Predicted preferences for all agents.
        """
        prefs_pred = []
        for i in range(self.N):
            x1s_i, x1s_minus_i = self.split_x(x1, i)
            x2s_i, x2s_minus_i = self.split_x(x2, i)
            fA = self.fc[i](self.th[i], x1s_i, x1s_minus_i)
            fB = self.fc[i](self.th[i], x2s_i, x2s_minus_i)
            pref = 1. if fA <= fB else 0.
            prefs_pred.append(pref)
        return prefs_pred

    def accuracy_score(self, ds: DataSet) -> float:
        """
        Compute the accuracy of the learned surrogate models on the provided preference data.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.

        Returns
        -------
        float
            Accuracy score.
        """
        prefs_pred = self.predict_dataset(ds)
        correct = 0
        total = 0
        for i in range(self.N):
            correct += jnp.sum(prefs_pred[i] == ds.prefs[i])
            total += ds.prefs[i].shape[0]
        accuracy = correct / total
        return accuracy

    def solve_gnep(self, x0: ArrayLike = None, J: callable = None, alpha1: float = None, alpha2: float = None,
                   pmin: ArrayLike = None, pmax: ArrayLike = None, f_gnep: list[callable] = None,
                   solver: str = "trf", verbose: int = 0, tol: float = 1e-12, max_nfev: int = 200) -> SimpleNamespace:
        """
        Solve the GNEP whose cost functions are given by self.fc with parameters self.th.
        The GNEP is solved using nashopt.GNEP.
        If self._parametric is True, then we solve the GNEP using nashopt.ParametricGNEP instead.

        Parameters
        ----------
        x0 : ArrayLike, optional
            Initial guess for GNE. Only used for non-parametric GNEP. If None, defaults to a vector of zeros.
        J : callable, optional
            Objective function of parametric GNEP, with signature J(x, p). Used if self._parametric is True.
        alpha1 : float, optional
            1-norm regularization parameter for state in parametric GNEP. Overrides self.alpha1 if provided.
            Used when self._parametric is True.
        alpha2 : float, optional
            2-norm regularization parameter for state in parametric GNEP. Overrides self.alpha2 if provided.
            Used when self._parametric is True.
        pmin : ArrayLike, optional
            Minimum bounds for the parameters in parametric GNEP. Overrides self.pmin if provided.
            Used when self._parametric is True.
        pmax : ArrayLike, optional
            Maximum bounds for the parameters in parametric GNEP. Overrides self.pmax if provided.
            Used when self._parametric is True.
        f_gnep : list of callable, optional
            List of cost functions for all players, with signature f_i(x, p=0.0).
            If None, surrogate functions self.fc with parameters self.th are used.
        solver : str, optional
            Solver to use for to solve the GNEP using nashopt.GNEP.solve(). Default is "trf".
            Used if self._parametric is False.
        verbose : int, optional
            Verbosity level. Default is 0.
        tol : float, optional
            Tolerance for the GNEP solver. Default is 1e-12.
        max_nfev : int, optional
            Maximum number of function evaluations for the GNEP solver. Default is 200.

        Returns
        -------
        SimpleNamespace
            Output of GNEP.solve() or ParametricGNEP.solve().
            Also contains msg_error, a string with any error messages generated during solving.
        """
        if x0 is None:
            x0 = jnp.zeros(self.dim)
        if J is None:
            J = lambda x, p: jnp.sum(p**2)  # Quadratic cost on parameter to regularize to 0.0 by default
        _alpha1 = alpha1 if alpha1 is not None else self.alpha1
        _alpha2 = alpha2 if alpha2 is not None else self.alpha2
        _pmin = pmin if pmin is not None else self.pmin
        _pmax = pmax if pmax is not None else self.pmax

        if f_gnep is None:
            f_gnep = self._get_f_gnep()

        if self._parametric:
            gnep = ParametricGNEP(sizes=self.sizes, npar=self.npar, f=f_gnep, g=self.g, ng=self.ng,
                                  h=self.h, nh=self.nh, lb=self.lb, ub=self.ub, Aeq=self.Aeq, beq=self.beq)
            sol_gnep = gnep.solve(J=J, pmin=_pmin, pmax=_pmax, alpha1=_alpha1, alpha2=_alpha2, verbose=verbose)
        else:
            gnep = GNEP(sizes=self.sizes, f=f_gnep, g=self.g, ng=self.ng,
                        h=self.h, nh=self.nh, lb=self.lb, ub=self.ub, Aeq=self.Aeq, beq=self.beq)
            try:
                sol_gnep = gnep.solve(x0=x0, solver=solver, verbose=verbose, tol=tol, max_nfev=max_nfev)
                msg_error = ""
            except Exception as e:
                new_solver = "lm" if solver == "trf" else "trf"
                msg_error = f"Error occurred while solving GNEP with solver '{solver}': '{e}'."
                msg_error += f" Switching to solver '{new_solver}' and trying again..."
                if verbose > 0:
                    print(msg_error)
                sol_gnep = gnep.solve(x0=x0, solver=new_solver, verbose=verbose, tol=tol, max_nfev=max_nfev)
        sol_gnep.msg_error = msg_error
        return sol_gnep

    def best_response(self, i: int, x_star: ArrayLike, rho: float = 1e5,
                      maxiter: int = 200, tol: float = 1e-8) -> SimpleNamespace:
        """
        Compute the best response of agent i given the strategies of the other agents.

        Parameters
        ----------
        i : int
            Index of the agent.
        x_star : ArrayLike
            Current strategies of all agents.
        rho : float, optional
            Penalty parameter used in nashopt.GNEP.best_response() to handle constraints. Default is 1e5.
        maxiter : int, optional
            Maximum number of L-BFGS-B iterations in nashopt.GNEP.best_response(). Default is 200.
        tol : float, optional
            Tolerance for the L-BFGS-B solver in nashopt.GNEP.best_response(). Default is 1e-8.

        Returns
        -------
        SimpleNamespace
            Output of nashopt.GNEP.best_response()
        """
        f_gnep = self._get_f_gnep()
        gnep = GNEP(sizes=self.sizes, f=f_gnep, g=self.g, ng=self.ng, h=self.h, nh=self.nh,
                    lb=self.lb, ub=self.ub, Aeq=self.Aeq, beq=self.beq)
        sol_br = gnep.best_response(i, x_star, rho=rho, maxiter=maxiter, tol=tol)
        return sol_br

    def x_i(self, x: ArrayLike, i: int) -> ArrayLike:
        """ Get player i's decision variables from the concatenated decision variable vector """
        return jnp.array(x)[jnp.array(self.idx_i[i])]

    def x_mi(self, x: ArrayLike, i: int) -> ArrayLike:
        """ Get the $x_{-i}$ decision variables from the concatenated decision variable vector """
        return jnp.array(x)[jnp.array(self.idx_minus_i[i])]

    def split_x(self, x: ArrayLike, i) -> tuple[ArrayLike, ArrayLike]:
        """ Split the concatenated decision variable vector into player i's variables and the rest """
        return self.x_i(x, i), self.x_mi(x, i)

    def join_i(self, x_i: ArrayLike, x_minus_i: ArrayLike, i: int) -> ArrayLike:
        """ Join player i's decision variables into the full decision variable vector """
        x_full = jnp.zeros(self.dim)
        x_full = jnp.array(x_full).at[jnp.array(self.idx_i[i])].set(x_i)
        x_full = jnp.array(x_full).at[jnp.array(self.idx_minus_i[i])].set(x_minus_i)
        return x_full

    def split_x_all(self, x: ArrayLike) -> list[ArrayLike]:
        """ Split the decision variables x into a list with the elements of each agent [x_1, ... x_N] """
        x_list = []
        for i in range(self.N):
            x_list.append(self.x_i(x, i))
        return x_list

    def join_x_all(self, x_list: list[ArrayLike]) -> ArrayLike:
        """ Joins a list [x_1, ... x_N] into a single array of decision variables x """
        x_full = jnp.zeros(self.dim)
        for i in range(self.N):
            x_full = jnp.array(x_full).at[jnp.array(self.idx_i[i])].set(x_list[i])
        return x_full

    def substitute_i(self, x: ArrayLike, x_i_new: ArrayLike, i: int) -> ArrayLike:
        """ Substitute player i's decision variables in the full decision variable vector with new values """
        x_new = jnp.array(x).at[jnp.array(self.idx_i[i])].set(x_i_new)
        return x_new

    def _get_f_gnep(self):
        """
        Returns the list of surrogate functions for all players using signature f_i(x, p).
        Required for interfacing with the GNEP solvers in nashopt, as it expects functions with this signature.
        """
        f_gnep = []
        for i in range(self.N):
            @jax.jit
            def fi(x, p=0.0, i=i):
                x_i = x[jnp.array(self.idx_i[i])]
                x_minus_i = x[jnp.array(self.idx_minus_i[i])]
                return self.fc[i](self.th[i], x_i, x_minus_i, p)
            f_gnep.append(fi)
        return f_gnep

    def _proba_pref(self, fA: float, fB: float, dist: float) -> float:
        """
        Compute the probability of preferring option A over B given their surrogate function values fA and fB, and
        the distance-based weight dist. """
        return 1. / (1. + jnp.exp((fA - fB) / (dist)))

    def project_to_feasible(self, x: ArrayLike, rho: float = 1e5,
                            maxiter: int = 200, tol: float = 1e-12) -> SimpleNamespace:
        """
        Project x to the feasible set defined by g(x) <= 0, h(x)= 0, Aeq x = beq and lb <= x <= ub.

        The g(), h(), Aeq, beq, lb, ub are defined in the properties of the PrefGNEP class.
        Projection problem is solved using SciPy's L-BFGS-B solver (ScipyBoundedMinimize in jaxopt).

        Parameters
        ----------
        x : ArrayLike
            Point to be projected.
        rho : float, optional
            Penalty parameter for the constraint violations in the projection optimization problem.
        maxiter : int, optional
            Maximum number of iterations for the optimization solver.
        tol : float, optional
            Tolerance for the optimization solver.

        Returns
        -------
        sol : SimpleNamespace
            A SimpleNamespace containing the projection sol.x and optimization statistics sol.stats.
        """

        t_start = timer()
        x = jnp.asarray(x)

        @jax.jit
        def fun(z, x=x):
            val = 0.5 * jnp.sum((z - x)**2)
            if self.g is not None:
                val += rho * jnp.sum(jnp.maximum(0.0, self.g(z))**2)
            if self.h is not None:
                val += rho * jnp.sum((self.h(z))**2)
            if self.Aeq is not None and self.beq is not None:
                val += rho * jnp.sum((self.Aeq @ z - self.beq)**2)
            return val

        options = {'iprint': -1, 'maxls': 20, 'gtol': tol, 'eps': tol,
                   'ftol': tol, 'maxfun': maxiter, 'maxcor': 10}

        solver = jaxopt.ScipyBoundedMinimize(
            fun=fun, tol=tol, method="L-BFGS-B", maxiter=maxiter, options=options)
        x_proj, state = solver.run(x, bounds=(self.lb, self.ub))
        t_end = timer()

        stats = SimpleNamespace()
        stats.elapsed_time = t_end - t_start
        stats.solver = state
        stats.iters = state.iter_num

        sol = SimpleNamespace()
        sol.x = x_proj
        sol.stats = stats
        return sol

    def project_i_to_feasible(self, i: int, x: ArrayLike, rho: float = 1e5,
                              maxiter: int = 200, tol: float = 1e-8) -> SimpleNamespace:
        """
        Project player i variables x_i to the feasible set, given other players' variables x_minus_i (which are kept fixed).
        Projection problem is solved using SciPy's L-BFGS-B solver (ScipyBoundedMinimize in jaxopt).

        Parameters
        ----------
        i : int
            Index of the player whose variables we want to project.
        x : ArrayLike
            Point to be projected. x = (x_i, x_minus_i). Point x_i is projected while keeping x_minus_i fixed.
        rho : float, optional
            Penalty parameter for the constraint violations in the projection optimization problem.
        maxiter : int, optional
            Maximum number of iterations for the optimization solver.
        tol : float, optional
            Tolerance for the optimization solver.

        Returns
        -------
        sol : SimpleNamespace
            A SimpleNamespace containing the projection sol.x and optimization statistics sol.stats.
        """
        t_start = timer()
        x = jnp.asarray(x)
        x_i, x_minus_i = self.split_x(x, i)
        lb_i = self.x_i(self.lb, i)
        ub_i = self.x_i(self.ub, i)

        def x_full(x_i, x_minus_i):
            return self.join_i(x_i, x_minus_i, i=i)

        @jax.jit
        def fun(z_i, x_i=x_i, x_mi=x_minus_i):
            z = x_full(z_i, x_mi)
            val = 0.5 * jnp.sum((z_i - x_i)**2)
            if self.g is not None:
                val += rho * jnp.sum(jnp.maximum(0.0, self.g(z))**2)
            if self.h is not None:
                val += rho * jnp.sum((self.h(z))**2)
            if self.Aeq is not None and self.beq is not None:
                val += rho * jnp.sum((self.Aeq @ z - self.beq)**2)
            return val

        options = {'iprint': -1, 'maxls': 20, 'gtol': tol, 'eps': tol,
                   'ftol': tol, 'maxfun': maxiter, 'maxcor': 10}

        solver = jaxopt.ScipyBoundedMinimize(
            fun=fun, tol=tol, method="L-BFGS-B", maxiter=maxiter, options=options)
        x_i_proj, state = solver.run(x_i, bounds=(lb_i, ub_i))
        t_end = timer()

        stats = SimpleNamespace()
        stats.elapsed_time = t_end - t_start
        stats.solver = state
        stats.iters = state.iter_num

        sol = SimpleNamespace()
        sol.x = self.join_i(x_i_proj, x_minus_i, i=i)
        sol.stats = stats
        return sol

    def oracle_pair(self, xA: ArrayLike, xB: ArrayLike, f_i: list[callable]) -> list[float]:
        """
        Oracle that returns preference between xA and xB for all players based on the values of functions f_i.

        Parameters
        ----------
        xA : ArrayLike
            Decision variables for option A.
        xB : ArrayLike
            Decision variables for option B.
        f_i : list of callable
            Functions that indicate the players' preferences (smaller value is preferred).

        Returns
        -------
        list of float
            Preferences for all agents: 1. if xA is preferred, 0. if xB is preferred.
        """
        prefs = []
        for i in range(self.N):
            fA = f_i[i](xA)
            fB = f_i[i](xB)
            pref = 1. if fA <= fB else 0.
            prefs.append(pref)
        return prefs

    def oracle_preference(self, i: int, xA: ArrayLike, xB: ArrayLike, x_all: ArrayLike, f_i: callable) -> float:
        """
        Oracle that returns preference between xA and xB for player i based on the values of functions f_i.

        Parameters
        ----------
        i : int
            Index of the player.
        xA : ArrayLike
            Decision variables for option A.
        xB : ArrayLike
            Decision variables for option B.
        x_all : ArrayLike
            Current decision variables of all players (value of x_i for player i is not used, as we substitute
            it with xA and xB).
        f_i : callable
            Function that indicates the player's preference (smaller value is preferred).

        Returns
        -------
        float
            1. if xA is preferred, 0. if xB is preferred.
        """
        x_all_A = self.substitute_i(x_all, xA, i)
        x_all_B = self.substitute_i(x_all, xB, i)
        f_A = f_i(x_all_A)
        f_B = f_i(x_all_B)
        return 1. if f_A <= f_B else 0.

    def oracle_dataset_at(self, ds: DataSet, j: int, f_i: list[callable]) -> list[float]:
        """
        Oracle that returns preferences for the j-th sample in the dataset for all players.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.
        j : int
            Index of the sample in the dataset.
        f_i : list of callable
            List of functions that indicate the players' preferences (smaller value is preferred).

        Returns
        -------
        list of float
            Preferences for all agents for the j-th sample in the dataset.
        """
        prefs = []
        for i in range(self.N):
            x_A = ds.xA[i][j]
            x_B = ds.xB[i][j]
            x_all = ds.samples[j]
            pref_i = self.oracle_preference(i, x_A, x_B, x_all, f_i[i])
            prefs.append(pref_i)
        return prefs

    def eval_dataset_at(self, ds: DataSet, j: int) -> list[float]:
        """
        Evaluate the surrogate cost functions self.fc for the j-th sample in the dataset for all players.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.
        j : int
            Index of the sample in the dataset.

        Returns
        -------
        tuple
            A tuple (prefs, f_A, f_B) where:
            - prefs : list of float
                Preferences for all agents for the j-th sample in the dataset.
            - f_A : list of float
                Evaluations of the surrogate cost functions for option A for all agents.
            - f_B : list of float
                Evaluations of the surrogate cost functions for option B for all agents.
        """
        f_A = []
        f_B = []
        prefs = []
        for i in range(self.N):
            x_A = ds.xA[i][j]
            x_B = ds.xB[i][j]
            x_all = ds.samples[j]
            val_A = self.fc[i](self.th[i], x_A, self.x_mi(x_all, i))
            f_A.append(val_A)
            val_B = self.fc[i](self.th[i], x_B, self.x_mi(x_all, i))
            f_B.append(val_B)
            prefs.append(1. if val_A <= val_B else 0.)
        return prefs, f_A, f_B

    def predict_proba_i(self, i: int, xA: ArrayLike, xB: ArrayLike, x_all: ArrayLike,
                        dist: Callable | float = dist_registry["log_inf"], dist_weight: float = 1.0) -> float:
        """
        Predict the probability that player i prefers x_A over x_B according to the
        surrogate model (using self.fc and self.th).

        Parameters
        ----------
        i : int
            Index of the player.
        xA : ArrayLike
            Decision variables for option A.
        xB : ArrayLike
            Decision variables for option B.
        x_all : ArrayLike
            Current decision variables of all players.
        dist : callable | float
            Distance-based weight for the probability calculation. Either a float or a callable function
            with signature dist(x_all, xA, xB) that computes the distance-based weight given the current
            decision variables of all players and the two options. Default is dist_registry["log_inf"].
        dist_weight : float
            Weight to apply to the distance-based weights.

        Returns
        -------
        float
            Predicted probability that player i prefers x_A over x_B.
        """
        fA = self.fc[i](self.th[i], xA, self.x_mi(x_all, i))
        fB = self.fc[i](self.th[i], xB, self.x_mi(x_all, i))
        if callable(dist):
            dist_xi = dist_weight * dist(x_all, xA, xB)
        elif dist is None:
            dist_xi = 1.0
        else:
            dist_xi = dist
        P = 1. / (1. + jnp.exp(1. * (fA - fB) / (dist_xi)))
        return P

    def predict_proba_dataset_at(self, ds: DataSet, j: int) -> list[float]:
        """
        Predict the probabilities that player i prefers xA over xB for the j-th sample in the dataset.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.
        j : int
            Index of the sample in the dataset.

        Returns
        -------
        list of float
            Predicted probabilities that each player prefers xA over xB for the j-th sample in the dataset.
        """
        proba = []
        for i in range(self.N):
            xA = ds.xA[i][j]
            xB = ds.xB[i][j]
            x_all = ds.samples[j]
            dist = ds.dist[i][j]
            proba_i = self.predict_proba_i(i, xA, xB, x_all, dist=dist)
            proba.append(proba_i)
        return proba

    def generate_initial_dataset(self, f, n_samples: int = 10, delta: float = 0.1,
                                 dist_metric: str | None = None, dist_weight: float | None = None) -> DataSet:
        """
        Generate a dataset of decision variables and preferences for all players.

        Parameters
        ----------
        f : list of callable
            Functions used to determine the preferences for each player.
        n_samples : int, optional
            Number of samples to generate. Default is 10.
        delta : float, optional
            Magnitude of perturbation for generating preference pairs. Default is 0.1.
        dist_metric : str or None, optional
            Distance metric to use for weighting the samples in the dataset. Defaults to DataSet default.
        dist_weight : float or None, optional
            Weight to apply to the distance-based weights in the dataset. Defaults to DataSet default.

        Returns
        -------
        DataSet
            Dataset containing the generated samples, preference pairs, and preference labels.
        """
        if self.lb is None or self.ub is None:
            raise ValueError("generate_initial_dataset() requires compact box constraints, but lb or ub is None.")
        if jnp.any(jnp.isinf(jnp.asarray(self.lb))):
            raise ValueError("generate_initial_dataset() requires compact box constraints, but lb contains -inf.")
        if jnp.any(jnp.isinf(jnp.asarray(self.ub))):
            raise ValueError("generate_initial_dataset() requires compact box constraints, but ub contains inf.")

        # Sample uniformly in lb and ub
        X_init_temp = jax.random.uniform(jax.random.PRNGKey(0), (n_samples, self.dim), minval=self.lb, maxval=self.ub)
        X_init = jnp.array([self.project_to_feasible(x).x for x in X_init_temp])

        X_pref_A = [[] for _ in range(self.N)]
        X_pref_B = [[] for _ in range(self.N)]
        prefs = [[] for _ in range(self.N)]

        for i in tqdm(range(self.N), desc="Generating preferences for players", leave=True):
            for j in tqdm(range(n_samples), leave=False, desc="Samples"):
                # Sample a random direction for player i
                key = jax.random.PRNGKey(i * n_samples + j)
                direction = jax.random.normal(key, (self.sizes[i],))
                direction = direction / jnp.linalg.norm(direction)
                x_i, _ = self.split_x(X_init[j], i)

                # Add perturbation in opposite directions and project back to feasible set to create two options for player i
                x_A_i = x_i + delta * direction
                x_B_i = x_i - delta * direction
                x_A_full = self.substitute_i(X_init[j], x_A_i, i)
                x_B_full = self.substitute_i(X_init[j], x_B_i, i)
                x_A_proj = self.project_i_to_feasible(i, x_A_full).x
                x_B_proj = self.project_i_to_feasible(i, x_B_full).x
                x_A, _ = self.split_x(x_A_proj, i)
                x_B, _ = self.split_x(x_B_proj, i)

                # Append options and preference label
                X_pref_A[i].append(np.array(x_A))
                X_pref_B[i].append(np.array(x_B))
                prefs[i].append(self.oracle_preference(i, x_A, x_B, X_init[j], f[i]))

        # Convert all outputs to jax.numpy arrays
        X_init = jnp.array(X_init)
        X_pref_A = [jnp.array(arr) for arr in X_pref_A]
        X_pref_B = [jnp.array(arr) for arr in X_pref_B]
        prefs = [jnp.array(arr) for arr in prefs]

        return DataSet(X_init, X_pref_A, X_pref_B, prefs, dist_metric=dist_metric, dist_weight=dist_weight)

    def check_sample_feasibility(self, X_samples: ArrayLike, eps_tol: float = 1e-3, assert_on_fail: bool = False) -> bool:
        """
        Check if the given samples satisfy the constraints.

        Parameters
        ----------
        X_samples : ArrayLike
            Array of decision variables.
        eps_tol : float, optional
            Tolerance for constraint satisfaction. Default is 1e-3.
        assert_on_fail : bool, optional
            If True, raises an assertion error if a sample is infeasible. Default is False.

        Returns
        -------
        bool
            True if all samples satisfy the constraints, False otherwise.
        """
        X_to_test = jnp.atleast_2d(X_samples)
        for i in range(X_to_test.shape[0]):
            _bound_lb = X_to_test[i] >= self.lb - eps_tol
            _bound_ub = X_to_test[i] <= self.ub + eps_tol
            if self.g is not None:
                _ineq = self.g(X_to_test[i]) <= eps_tol
            else:
                _ineq = np.array([True])
            if self.h is not None:
                _eq_h = jnp.allclose(self.h(X_to_test[i]), jnp.zeros(self.nh), atol=eps_tol)
            else:
                _eq_h = True
            if self.Aeq is not None and self.beq is not None:
                _eq = jnp.allclose(self.Aeq @ X_to_test[i], self.beq, atol=eps_tol)
            else:
                _eq = True
            if assert_on_fail:
                assert all(_ineq), f"g(x) <= 0 not satisfied for sample {i}"
                assert all(_bound_lb) and all(_bound_ub), f"lb <= x <= ub not satisfied for sample {i}"
                assert _eq, f"Aeq x = beq not satisfied for sample {i}"
                assert _eq_h, f"h(x) = 0 not satisfied for sample {i}"
            else:
                if not (all(_ineq) and all(_bound_lb) and all(_bound_ub) and _eq and _eq_h):
                    return False
        return True

    def check_dataset_feasibility(self, ds: DataSet, assert_on_fail: bool = False, verbose: int = 0) -> bool:
        """
        Check if samples in the dataset satisfy the constraints.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.
        assert_on_fail : bool, optional
            If True, raises an assertion error if a sample is infeasible. Default is False.
        verbose : int, optional
            Verbosity level. Default is 0.

        Returns
        -------
        bool
            True if all samples satisfy the constraints, False otherwise.
        """
        check_msg = ""
        for i in range(self.N):
            for j in range(ds.xA[i].shape[0]):
                x_A_full = self.substitute_i(ds.samples[j], ds.xA[i][j], i)
                x_B_full = self.substitute_i(ds.samples[j], ds.xB[i][j], i)
                feas_A = self.check_sample_feasibility(x_A_full[jnp.newaxis, :], assert_on_fail=False)
                feas_B = self.check_sample_feasibility(x_B_full[jnp.newaxis, :], assert_on_fail=False)
                if not feas_A:
                    check_msg += f"Sample {j} for player {i} option A is infeasible: {x_A_full}\n"
                if not feas_B:
                    check_msg += f"Sample {j} for player {i} option B is infeasible: {x_B_full}\n"
        if check_msg != "":
            if assert_on_fail:
                raise AssertionError(check_msg)
            elif verbose > 0:
                print(check_msg)
                return False
        return True

    def eval_f_i(self, i: int, x: ArrayLike) -> float:
        """
        Evaluate the surrogate cost function for player i at point x.

        Parameters
        ----------
        i : int
            Index of the player.
        x : ArrayLike
            Decision variable vector.

        Returns
        -------
        float
            Evaluated cost for agent i at point x.
        """
        x_i, x_minus_i = self.split_x(x, i)
        return self.fc[i](self.th[i], x_i, x_minus_i)

    def augment_dataset_AL(self, ds: DataSet, f, delta: float = 1.0, sigma: float = 1e-1, x0: ArrayLike = None,
                           space_fill: bool = False, ps_iters: int = 100) -> DataSet:
        """
        Augments the dataset by adding a new sample that balances exploration and exploitation.

        Exploitation is done by considering the current GNEP solution, while exploration is done by
        modifying the current surrogate models to search in unexplored regions.

        Parameters
        ----------
        ds : DataSet
            Dataset containing samples, preference pairs, and true preference labels.
        f : list of callable
            Functions used to determine the preferences.
        delta : float, optional
            Parameter that determines the relative weight of the exploration term. Default is 1.0.
        sigma : float, optional
            Determines magnitude of noise to add to previous xA when creating new xB. Default is 1e-1.
        x0 : ArrayLike, optional
            Initial guess for GNEP solver.
        space_fill : bool, optional
            Whether to use space-filling for exploration. If False, random sampling is used. Default is False.
        ps_iters : int, optional
            Number of iterations for the particle swarm optimization when using space-filling. Default is 100.

        Returns
        -------
        DataSet
            Updated dataset with new sample and preference.
        """
        if self.lb is None or self.ub is None:
            raise ValueError("augment_dataset_AL() only supports compact box constraints, but lb or ub is None.")
        if jnp.any(jnp.isinf(jnp.asarray(self.lb))):
            raise ValueError("augment_dataset_AL() only supports compact box constraints, but lb contains -inf.")
        if jnp.any(jnp.isinf(jnp.asarray(self.ub))):
            raise ValueError("augment_dataset_AL() only supports compact box constraints, but ub contains inf.")

        if not space_fill:  # Option 1: Use a random xi for each player within bounds
            new_xi = []
            for i in range(self.N):
                new_xi.append(jax.random.uniform(jax.random.PRNGKey(ds.size * self.N + i),
                                                 (self.sizes[i],),
                                                 minval=self.x_i(self.lb, i),
                                                 maxval=self.x_i(self.ub, i)))
        else:  # Option 2: Use space-filling to select new xi for each player

            new_xi = []
            for i in range(self.N):
                ps_solver = ps.single.GlobalBestPSO(n_particles=50, dimensions=self.sizes[i], options=self._opt_ps,
                                                    bounds=(self.x_i(self.lb, i), self.x_i(self.ub, i)))

                @jax.jit
                def _min_dist(x):
                    dist = jax.vmap(lambda other_x: jnp.sum((x - other_x) ** 2))(ds.samples[:, jnp.array(self.idx_i[i])])
                    return jnp.min(dist)

                @jax.jit
                def space_fill_obj(x):
                    return -_min_dist(x)

                _, new_x0 = ps_solver.optimize(space_fill_obj, iters=ps_iters, verbose=0)
                new_xi.append(new_x0)

        # Surrogate functions with exploration term f_new_i(x, p) = f_i(x_i, x_minus_i, p) + delta * ||x_i - new_xi||^2
        f_gnep = self._get_f_gnep()
        f_new = []
        for i in range(self.N):
            @jax.jit
            def fi(x, p=0.0, i=i):
                return f_gnep[i](x, p) + delta * jnp.sum((self.x_i(x, i) - new_xi[i])**2)
            f_new.append(fi)

        # Solve GNEP problem using these exloration-augmented surrogate functions
        info = NamedTuple('GNEPinfo', [('res_gnep', float), ('res_br', list[float]), ('msg_error', str)])
        new_x = self.solve_gnep(x0, f_gnep=f_new)
        info.res_gnep = float(jnp.linalg.norm(new_x.res, ord=np.inf))
        info.msg_error = new_x.msg_error

        # Add new sample to dataset
        new_sample = new_x.x
        new_xA = []
        new_xB = []
        new_prefs = []
        info.res_br = []
        for i in range(self.N):
            # Choose new xA as the GNEP solution found (with exploration term)
            new_xA.append(self.x_i(new_x.x, i))
            # Choose new xB as the best response to new_x plus some noise
            sol_br_i = self.best_response(i, new_x.x)
            info.res_br.append(sol_br_i.stats.solver.success)
            new_xB_noise = sigma * (np.random.rand(1) - 0.5) * np.abs(self.x_i(sol_br_i.x, i))
            new_xB_noise = self.x_i(sol_br_i.x, i) + new_xB_noise
            # Clip new_xB to be within bounds
            lb_i = self.x_i(self.lb, i)
            ub_i = self.x_i(self.ub, i)
            new_xB.append(jnp.clip(new_xB_noise, lb_i, ub_i))  # NOTE: Not forcing feasibility, just clipping to local bounds
            new_prefs.append(self.oracle_preference(i, new_xA[-1], new_xB[-1], new_sample, f[i]))
        ds.add_sample(new_sample, new_xA, new_xB, new_prefs)

        return ds, info

    def fit_AL_loop(self, f: list[callable], ds: DataSet, n_iters: int, x0: ArrayLike = None,
                    delta: float = 1.0, p_delta: float = 5., delta_min: float = 1e-3,
                    sigma: float = 1e-1, p_sigma: float = 4., sigma_min: float = 1e-3,
                    f_eval: callable = None, tol_eval: float = 0.0, tol_gnep: float = 0.0, tol_res: float = 1e-4,
                    n_reset_expl: int = None, n_reset_noise: int = None, update_th_0: bool = False,
                    store_gnep_sol: bool = False, store_accuracy: bool = False, store_best_th: bool = True,
                    min_iters: int = 0, mu_th: float = 0.0, it_add_mu: int = None,
                    hist: SimpleNamespace = None, space_fill: bool = False, ps_iters: int = 100, verbose: int = 1,
                    seed: int | None = None) -> tuple[ArrayLike, list[ArrayLike], list[ArrayLike], list[ArrayLike], dict]:
        """
        Active learning loop to fit the surrogate functions. At each iteration, the dataset is augmented using an
        exploitation-exploration strategy. The surrogate functions are refit at each iteration using the updated dataset,
        and the GNEP problem is solved using the updated surrogate functions.
        The objective is to converge to an approximate GNE of the original GNEP problem.

        Parameters
        ----------
        f : list of callable
            Functions used to determine the preferences for each player.
        ds : DataSet
            Initial dataset containing samples, preference pairs, and true preference labels.
        n_iters : int
            Number of iterations to perform.
        x0 : ArrayLike, optional
            Initial guess for GNEP solver. If None, defaults to a vector of zeros.
        delta : float, optional
            Magnitude of the exploration term added to the surrogate functions. Default is 1.0.
        p_delta : float, optional
            Exponent for decreasing delta over iterations. Default is 5.0.
        delta_min : float, optional
            Minimum value for delta to ensure some level of exploration is maintained. Default is 1e-3.
        sigma : float, optional
            Magnitude of noise to add when generating new preference pairs. Default is 1e-1.
        p_sigma : float, optional
            Exponent for decreasing sigma over iterations. Default is 4.0.
        sigma_min : float, optional
            Minimum noise level. Default is 1e-3.
        f_eval : callable, optional
            Optional function to evaluate at the GNEP solution for monitoring and stopping criterion.
            The evaluation is computed as eval = f_eval(x_star) and stored in the history at each iteration.
            eval is expected to be an array of dim=self.N, where eval[i] is the evaluation for player i.
        tol_eval : float, optional
            Tolerance for stopping criterion based on the inf-norm of f_eval at the GNEP solution. Default is 0.0 (disabled).
        tol_gnep : float, optional
            Tolerance for the relative change in GNEP solution used as a stopping criterion. Default is 0.0 (disabled).
        tol_res : float, optional
            Maximum allowed residual for GNEP or best response solutions before a warning is raised. Default is 1e-4.
        n_reset_expl : int, optional
            Number of iterations after which to reset the exploration parameter delta. If None, no reset.
        n_reset_noise : int, optional
            Number of iterations after which to reset the noise level sigma. If None, no reset.
        update_th_0 : bool, optional
            Whether to update the initial condition of the learning problem at each AL iteration using the current
            surrogate model parameters self.th. Default is False.
        store_gnep_sol : bool, optional
            Whether to compute and store the GNEP solution at each iteration. Default is False.
        store_accuracy : bool, optional
            Whether to compute and store the prediction accuracy on the dataset at each iteration. Default is False.
        store_best_th : bool, optional
            Whether to store the best surrogate model parameters, selected according to the minimum value of f_eval
            at the GNEP solution. Requires f_eval to be provided. Default is True.
        min_iters : int, optional
            Minimum number of iterations to perform before checking stopping criteria. Default is 0.
        mu_th : float, optional
            Regularization in the loss function for penalizing the distance to the current parameters. Default is 0.0.
        it_add_mu : int, optional
            Iteration at which to start adding the mu_th regularization term. If None, defaults to the last
            quarter of iterations (n_iters - n_iters // 4).
        hist : ALloopInfo, optional
            Optional history object to continue storing results from a previous run. If None, a new ALloopInfo
            object will be created.
        space_fill : bool, optional
            Whether to use space-filling (via particle swarm optimization) for exploration when augmenting the
            dataset. If False, random sampling is used. Default is False.
        ps_iters : int, optional
            Number of iterations for the particle swarm optimization when using space-filling. Default is 100.
        verbose : int, optional
            Verbosity level. Default is 1. Levels: 0 = no output, 1 = summary at each iteration, 2 = print detailed info.
        seed : int or None, optional
            Random seed for reproducibility. If None, no seed is set. Default is None.

        Returns
        -------
        ds : DataSet
            Final dataset after all active learning iterations.
        hist : ALloopInfo
            History of the AL loop, containing GNEP solutions, accuracy scores, evaluation metrics,
            stopping criterion values, and other diagnostics at each iteration.

        Usage
        -----
        Learned GNE can be obtained by running sol = PrefGNEP.solve_gnep(x0), with the x0 used as argument of this method.
        Learned GNE is then sol.x, and the residual of the GNEP solution (for the surrogate GNEP) is sol.res.
        """

        if seed is not None:
            np.random.seed(seed)

        if x0 is not None:
            x0 = jnp.array(x0)
            assert x0.shape == (self.dim,), f"Expected x0 to have shape ({self.dim},), but got {x0.shape}"
        else:
            x0 = jnp.zeros(self.dim)

        if hist is not None:
            assert isinstance(hist, ALloopInfo), "Expected hist to be an ALLoopInfo object"
            hist.to_list()  # Convert any arrays in hist to lists so that appending works
        else:
            hist = ALloopInfo()

        if n_reset_expl is not None:
            assert n_reset_expl > 0, "Expected n_reset_expl to be a positive integer"
        else:
            n_reset_expl = n_iters

        if n_reset_noise is not None:
            assert n_reset_noise > 0, "Expected n_reset_noise to be a positive integer"
        else:
            n_reset_noise = n_iters

        if it_add_mu is not None:
            assert it_add_mu >= 0, "Expected it_add_mu to be a non-negative integer"
        else:
            it_add_mu = n_iters - n_iters // 4  # Default: start adding mu_th in the last quarter of iterations

        assert delta >= 0.0, "Expected delta to be non-negative"
        assert sigma >= 0.0, "Expected sigma to be non-negative"
        assert sigma_min >= 0.0, "Expected sigma_min to be non-negative"
        assert p_delta >= 0.0, "Expected p_delta to be non-negative"
        assert p_sigma >= 0.0, "Expected p_sigma to be non-negative"
        assert tol_gnep >= 0.0, "Expected tol_gnep to be non-negative"
        assert tol_eval >= 0.0, "Expected tol_eval to be non-negative"
        assert min_iters >= 0, "Expected min_iters to be a non-negative integer"
        assert mu_th >= 0.0, "Expected mu_th to be non-negative"

        if verbose > 0:
            iters_tqdm = tqdm(total=n_iters, desc='AL loop', leave=True, position=0)
            info_log = tqdm(total=0, position=1, bar_format='{desc}')
            error_msg = ""  # We store warning or error messages and print them at the end to avoid messing with tqdm display

        th_0 = copy.deepcopy(self.th)

        should_exit = False
        for it in range(n_iters):

            delta_it = delta * (1. - (it % n_reset_expl) / n_reset_expl) ** p_delta
            delta_it = max(delta_it, delta_min)

            sigma_it = sigma * (1. - (it % n_reset_noise) / n_reset_noise) ** p_sigma
            sigma_it = max(sigma_it, sigma_min)

            _mu_th = 0.0 if it < it_add_mu else mu_th

            hist.delta.append(delta_it)
            hist.sigma.append(sigma_it)
            hist.mu_th.append(_mu_th)

            if update_th_0:
                th_0 = copy.deepcopy(self.th)

            _, _, stats = self.fit(ds, update=True, mu_th=_mu_th, th_0=th_0)

            ds, info_aug_ds = self.augment_dataset_AL(ds, f, delta=delta_it, sigma=sigma_it, x0=x0,
                                                      space_fill=space_fill, ps_iters=ps_iters)
            hist.res_ds_gnep.append(info_aug_ds.res_gnep)
            hist.res_ds_br.append(info_aug_ds.res_br)
            if info_aug_ds.msg_error != "":
                error_msg += f" > Iteration {it} (augment_dataset_AL): {info_aug_ds.msg_error}\n"

            # Compute data and stopping criteria
            exit_gnep = exit_eval = False
            hist.times.append(stats.time)
            best_eval_norm = np.inf
            if store_gnep_sol or tol_gnep > 0.0 or f_eval is not None:
                sol_gnep = self.solve_gnep(x0)
                hist.res_x_star.append(float(jnp.linalg.norm(sol_gnep.res, ord=np.inf)))
                hist.x_star.append(sol_gnep.x)
                if sol_gnep.msg_error != "":
                    error_msg += f" > Iteration {it} (x_star): {sol_gnep.msg_error}\n"
                if tol_gnep > 0.0 and it > 0:
                    norm_diff = jnp.linalg.norm(hist.x_star[-1] - hist.x_star[-2], ord=jnp.inf)
                    norm_pref = jnp.linalg.norm(hist.x_star[-2], ord=jnp.inf)
                    gnep_diff = norm_diff / (norm_pref + 1e-8)
                    if gnep_diff < tol_gnep:
                        exit_gnep = True
                    hist.tol_gnep.append(gnep_diff)
                if f_eval is not None:
                    f_val = f_eval(sol_gnep.x)
                    hist.eval.append(f_val)
                    eval_norm = np.linalg.norm(f_val, ord=np.inf)
                    hist.tol_eval.append(eval_norm)
                    if eval_norm < tol_eval:
                        exit_eval = True
                    if store_best_th and f_eval is not None:
                        if eval_norm < best_eval_norm:
                            best_eval_norm = eval_norm
                            hist._replace(best_th=copy.deepcopy(self.th))
            if store_accuracy:
                acc = self.accuracy_score(ds)
                hist.accuracy.append(acc)

            # Exit logic based on user-specified conditions
            should_exit = False
            if it >= min_iters:
                if tol_gnep > 0.0 and tol_eval > 0.0:
                    if exit_gnep and exit_eval:
                        should_exit = True
                elif tol_gnep > 0.0:
                    if exit_gnep:
                        should_exit = True
                elif tol_eval > 0.0:
                    if exit_eval:
                        should_exit = True

            if verbose > 0:
                str_info = f"delta: {delta_it:.4f}, sigma: {sigma_it:.4f}, mu: {_mu_th:.4f}"
                if store_accuracy:
                    str_info += f", acc: {acc:.4f}"
                color_ds_gnep = '\033[0m' if hist.res_ds_gnep[-1] <= tol_res else '\033[1;31m'
                str_info += f", res_ds_gnep: {color_ds_gnep}{hist.res_ds_gnep[-1]:1.1g}\033[0m"
                color_ds_br = '\033[0m' if all(hist.res_ds_br[-1]) else '\033[1;31m'
                str_info += f", res_ds_br: {color_ds_br}{all(hist.res_ds_br[-1])}\033[0m"
                if f_eval is not None:
                    str_info += f", ||f_eval||_inf: {np.linalg.norm(f_val, ord=np.inf):.4f}"
                info_log.set_description_str(str_info)
            if verbose > 0:
                iters_tqdm.update(1)

            if should_exit:
                break

        # Convert lists to arrays in hist for easier use later
        hist = hist.to_array()
        hist = hist.assign_n_iters()

        if verbose > 0:
            iters_tqdm.close()
            info_log.close()
            if n_iters == hist.n_iters:
                print("AL loop: Reached maximum number of iterations.")
            elif should_exit:
                print(f"AL loop: Stopping criterion reached at iteration {it + 1}")
                if tol_gnep > 0.0:
                    print(f"  GNEP solution change: {gnep_diff} < {tol_gnep}")
                if tol_eval > 0.0:
                    print(f"  Evaluation value: {eval_norm} < {tol_eval}")
            if any(hist.res_ds_gnep > tol_res):
                print(f"\033[1;33mWarning: Some GNEP dataset residual exceeded {tol_res}. Max was {np.max(hist.res_ds_gnep)}. See hist.res_ds_gnep.\033[0m")  # noqa: E501
            if not hist.res_ds_br.all():
                print("\033[1;33mWarning: At least one BR problem failed when generating dataset. See hist.res_ds_br.\033[0m")  # noqa: E501
            if verbose > 1 and error_msg != "":
                print(f"Errors or warnings during AL loop:\n{error_msg}")

        return ds, hist
