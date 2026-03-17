"""
Example: Preference-Based Nash Equilibrium Learning for LQR Games
=================================================================

This script demonstrates the use of preference-based learning to find a
Nash equilibrium in a Linear Quadratic Regulator (LQR) game with N agents.

The multi-agent LQR game is defined by:
    - A shared linear dynamical system:  x_{t+1} = A x_t + B u_t, with `nx` states.
      We have `N` agents, each controlling a subset of the input u_t.
      Each agent controls `nu` inputs, so the total input dimension is `nu * N`.
    - Each agent minimizes their own quadratic cost function:
            J_i(u_i, u_{-i}) = sum_t [ x_t^T Q_i x_t + u_{i,t}^T R_i u_{i,t} ]  (1)

It is well known that this problem admits a Nash equilibrium of the form
u_i = -K_Nash_i x_t, where K_Nash_i is state-feedback gain for agent i,
u = (u_1, ..., u_N) and K_Nash = (K_Nash_1, ..., K_Nash_N), see e.g. [1].

The PrefGNEP class is used to learn an approximate Nash equilibrium, i.e. feedback gains
K_pref that approximate the true Nash gains, from pairwise preferences over candidate
feedback-gain matrices, without direct access to the cost functions (1) or their gradients.

We compare the preference-based Nash solution against the exact Nash solution (computed via
the nashopt package [2]) by evaluating the closed-loop cost of each controller starting from
a set of test initial states.

Usage:
------
Set `problem_size` to 'small', 'medium', or 'large' to select predefined problem dimensions.
Plots can be automatically saved by setting `save_plots` to True.
Other hyperparameters related to the learning loop can be modified in the "Problem setup" section.

References:
-----------
[1] B.Nortmann, A.Monti, M.Sassano, and T.Mylvaganam, "Nash equilibria for linear quadratic
    discrete-time dynamic games via iterative and data-driven algorithms,” IEEE Transactions
    on Automatic Control, vol. 69, no. 10, pp. 6561–6575, 2024.

[2] A. Bemporad, “Nashopt - A Python library for computing generalized Nash equilibria,” arXiv
    preprint arXiv:2512.23636, 2025.

(c) 2026 Pablo Krupa
"""

import numpy as np
import jax.numpy as jnp
from nashopt import NashLQR
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import partial
import pickle
import copy
from tqdm import tqdm
import sys
sys.path.append('../src')
from pref_gnep import PrefGNEP  # noqa: E402
from dataset import DataSet  # noqa: E402
from models import gen_quad_models  # noqa: E402

seed = 1
np.random.seed(seed)

# %% #################################
# Problem setup
######################################

problem_size = "large"  # "small", "medium" or "large"
gen_dataset = True  # Whether to generate a new dataset or load existing one
save_plots = True  # Whether to save the plots
ds_size_init = 50  # Dimension of initial dataset
n_iters_AL = 200  # Number of iterations of the AL loop
delta = 5.0  # Exploration parameter in the AL loop
p_delta = 5.0  # Exponent for exploration parameter decay in the AL loop
delta_min = 1e-3  # Minimum value of exploration parameter in the AL loop
sigma = 0.3  # Noise parameter for the AL loop
p_sigma = 4.0  # Exponent for noise parameter decay in the AL loop
sigma_min = 1e-3  # Minimum value of noise parameter in the AL loop
n_tests = 100  # Number of closed-loop simulations to run for post-learning evaluation
dare_iters = 50  # number of iterations for DARE solver
Tsim = dare_iters  # number of closed-loop simulation steps

# Set problem size based on problem_size variable
if problem_size == "small":
    nx = 6   # number of states
    nu = 2   # number of inputs per agent
    N = 3    # number of agents
elif problem_size == "medium":
    nx = 8
    nu = 2
    N = 4
elif problem_size == "large":
    nx = 12
    nu = 3
    N = 4
else:
    raise ValueError(f"Invalid problem_size '{problem_size}'. Must be 'small', 'medium' or 'large'")

print(f"\nLQR problem '{problem_size}' selected. nx: {nx}, nu: {nu}, N: {N}")

nu_i = [nu] * N  # Number of inputs of each agent

# Random unstable dynamics
A = np.random.rand(nx, nx)  # random A, possibly unstable
A = A / max(abs(np.linalg.eigvals(A))) * 1.1  # scale to have spectral radius = 1.1
B = 0.1 * np.random.randn(nx, nu * N)
C = np.ones((1, nx))
ny = C.shape[0]  # Number of outputs

# Define the cost matrices for each agent
Q = []
R = []
not_i = []
for i in range(N):
    # LQR weights of each agent
    Qi = np.zeros((nx, nx))
    # Each agent penalizes the states according to its assigned indices in sizes
    start_idx = sum(nu_i[:i])
    end_idx = start_idx + nu_i[i]
    Qi[start_idx:end_idx, start_idx:end_idx] = np.eye(nu_i[i])
    Ri = 0.1 * np.eye(nu_i[i])
    Q.append(Qi)
    R.append(Ri)

R_all = np.block([
    [R[i] if i == j else np.zeros_like(R[0]) for j in range(N)]
    for i in range(N)
])

x0_test = np.random.randn(nx, n_tests)  # Initial stetes used for post-learning evaluation of the controllers

# %% #################################
# Solve using nashopt package
######################################

print("\n=== Finding GNE solution using nashopt ===")
nash_lqr = NashLQR(nu_i, A, B, Q, R, dare_iters=dare_iters)
sol = nash_lqr.solve(method='residual', verbose=2)

K_Nash = copy.deepcopy(sol.K_Nash)
residual = sol.residual
stats = sol.stats
K_cen = sol.K_centralized

res = float(jnp.linalg.norm(residual))
np.set_printoptions(precision=4, suppress=True)

if res < 1.e-6:
    print("> GNE solution found")
    # print(f"K_Nash = {K_Nash}")
else:
    print("> GNE solution NOT found")

print(f"\tKKT residual norm:   {res: 10.7g}")
print(f"\tKKT evaluations:     {int(stats.kkt_evals): 3d}")
print(f"\tElapsed time:        {stats.elapsed_time: .2f} seconds")

# %% #################################
# Solve problem using preferences
######################################

f_real = []
for i in range(N):
    f_real.append(partial(nash_lqr.lqr_fun, i=i, A=A, B=B, Q=Q, R=R))

lb = K_Nash.flatten() - np.max(np.abs(K_Nash)) * 1.5
ub = K_Nash.flatten() + np.max(np.abs(K_Nash)) * 1.5

sizes = [nu_i[i] * nx for i in range(N)]  # Dimension of each agent's decision variable (feedback gain matrix flattened)
dim = sum(sizes)

# Initialize PrefGNEP, with diagonal quadratic surrogate models
th_0, fc, th_min, th_max = gen_quad_models(sizes, full=False, diagonal=True, with_linear=False)
pref_gnep = PrefGNEP(sizes=sizes, fc=fc, lb=lb, ub=ub)
pref_gnep.init(th=th_0, th_min=th_min, th_max=th_max, alpha1=0.0, alpha2=0.01)
pref_gnep.loss(rho_th=0.001)
pref_gnep.optimization(adam_epochs=0)

# Generate initial dataset
if gen_dataset:
    print("\n=== Generating initial dataset and feasibility check ===")
    ds = pref_gnep.generate_initial_dataset(f_real, n_samples=ds_size_init, delta=0.1)
    with open("example_lqr_dataset.pkl", "wb") as temp_file:
        pickle.dump((ds.samples, ds.xA, ds.xB, ds.prefs), temp_file)
else:
    print("\n=== Loading initial dataset and feasibility check ===")
    with open("example_lqr_dataset.pkl", "rb") as temp_file:
        samples, xA, xB, prefs = pickle.load(temp_file)
        ds = DataSet(samples, xA, xB, prefs)

# Test that the dataset is feasible
samples_feasible = pref_gnep.check_sample_feasibility(ds.samples, assert_on_fail=False)
dataset_feasible = pref_gnep.check_dataset_feasibility(ds, assert_on_fail=False)
print(f"All samples feasible: {samples_feasible}")
print(f"All dataset feasible: {dataset_feasible}")

# Initial fit
print("\n=== Fit PrefGNEP using AL method ===")
th_fit, infos, stats = pref_gnep.fit(ds)
pred_th_fit = pref_gnep.predict_dataset(ds)
print(f"Accuracy with init dataset: {pref_gnep.accuracy_score(ds):.4f}")


def x_star_eval(x_star):
    # Used to evaluate the best-response deviation in the iterates of the AL loop.
    # Might not be available in a real application, but we use it here to provide some feedback
    # in stdout about how the iterates are doing in terms of approximating a Nash equilibrium.
    eval = []
    nash_lqr.K_Nash = x_star.reshape(nu * N, nx)
    for i in range(N):
        K_best_i = nash_lqr.dare(agent=i)
        K_iter_i = x_star.reshape(nu * N, nx)[nash_lqr.ii[i], :]
        eval.append(jnp.sum((K_best_i - K_iter_i)**2))  # Frobenius norm squared
    return jnp.array(eval)


# Learn the Nash equilibrium using the active learning loop
x0 = jnp.zeros(dim)
ds, hist_l = pref_gnep.fit_AL_loop(f_real, ds, n_iters=n_iters_AL, x0=x0,
                                   sigma=sigma, p_sigma=p_sigma, sigma_min=sigma_min,
                                   delta=delta, p_delta=p_delta, delta_min=delta_min,
                                   store_gnep_sol=True, store_accuracy=True, f_eval=x_star_eval,
                                   verbose=2, seed=seed, update_th_0=True)

# %% #################################
# Check and compare results
######################################

# Compute the Nash equilibrium with the learned parameters
sol_pref = pref_gnep.solve_gnep(x0)
K_pref = sol_pref.x.reshape(nu * N, nx)

# Compute best-response deviation at each iteration
br_devs = []
for i in range(N):
    br_devs.append([])
for x_iter in tqdm(hist_l.x_star, desc="Deviation from BR"):
    nash_lqr.K_Nash = x_iter.reshape(nu * N, nx)
    for i in range(N):
        K_best_i = nash_lqr.dare(agent=i)
        K_iter_i = x_iter.reshape(nu * N, nx)[nash_lqr.ii[i], :]
        br_devs[i].append(jnp.sum((K_best_i - K_iter_i)**2))  # Frobenius norm squared
for i in range(N):
    br_devs[i] = np.array(br_devs[i])

# Check stability of closed-loop system with the different controllers
rad_nash = np.abs(np.linalg.eigvals(A - B @ K_Nash)[0])
rad_cen = np.abs(np.linalg.eigvals(A - B @ K_cen)[0])
rad_pref = np.abs(np.linalg.eigvals(A - B @ K_pref)[0])
rad_ol = np.abs(np.linalg.eigvals(A)[0])
rad_loop = [[] for _ in range(hist_l.n_iters)]
for i in range(hist_l.n_iters):
    K_iter = hist_l.x_star[i].reshape(nu * N, nx)
    rad_loop[i] = np.abs(np.linalg.eigvals(A - B @ K_iter)[0])
print("\n\033[1;34mSpectral radius:\033[0m")
print(f"\033[1;32mopen-loop\033[0m system:                         {rad_ol:.4f}")
print(f"closed-loop system with \033[1;31mNash gains\033[0m:       {rad_nash:.4f}")
print(f"closed-loop system with \033[1;36mpreference-based\033[0m: {rad_pref:.4f}")
print(f"closed-loop system with \033[1;35mcentralized LQR\033[0m:  {rad_cen:.4f}")


def cl_cost(X, U):
    # Computes the closed-loop cost of a trajectory X, U according to the cost functions of the agents.
    cost = 0.0
    for i in range(N):
        start_idx = sum(nu_i[:i])
        end_idx = start_idx + nu_i[i]
        for t in range(len(U)):
            cost += X[t].T @ Q[i] @ X[t] + U[t][start_idx:end_idx].T @ R[i] @ U[t][start_idx:end_idx]
    return cost


def cl_loop(x0, K_cl, Tsim=20):
    # Simulates the closed-loop system with initial state x0 and feedback gain K_cl for Tsim steps,
    # and computes the cost of the trajectory.
    X = [x0]
    U = []
    Y = [C @ x0]
    for k in range(Tsim):
        uk = -K_cl @ X[-1]
        U.append(uk)
        x_next = A @ X[-1] + B @ uk
        X.append(x_next)
        Y.append(C @ x_next)
    cost = cl_cost(X, U)
    return cost, np.array(X), np.array(U), np.array(Y)


cost_cen = []  # Closed-loop cost with centralized LQR controller
cost_nash = []  # Closed-loop cost with true Nash controller
cost_pref = []  # Closed-loop cost with preference-based Nash controller
cost_loop = [[] for _ in range(hist_l.n_iters)]  # Closed-loop cost with iterates of the AL loop

for j in tqdm(range(n_tests), desc="Doing closed-loop sims", leave=True):
    x0_j = x0_test[:, j]
    cost_cen_j, X_cen, U_cen, Y_cen = cl_loop(x0_j, K_cen, Tsim)
    cost_pref_j, X_pref, U_pref, Y_pref = cl_loop(x0_j, K_pref, Tsim)
    cost_nash_j, X_nash, U_nash, Y_nash = cl_loop(x0_j, K_Nash, Tsim)
    cost_cen.append(cost_cen_j)
    cost_pref.append(cost_pref_j)
    cost_nash.append(cost_nash_j)
    for i in range(hist_l.n_iters):
        K_iter = hist_l.x_star[i].reshape(nu * N, nx)  # Skip first iter (initial guess)
        cost_loop_ij, _, _, _ = cl_loop(x0_j, K_iter, Tsim)
        cost_loop[i].append(cost_loop_ij)  # Clip cost to avoid outliers dominating the plot
cost_cen = np.array(cost_cen)
cost_pref = np.array(cost_pref)
cost_nash = np.array(cost_nash)
for i in range(hist_l.n_iters):
    cost_loop[i] = np.array(cost_loop[i])

# Compute RMSE of the costs of the learning loop w.r.t. the Nash cost
rmse_cost_pref = np.sqrt(np.mean((cost_pref - cost_nash)**2))
rmse_cost_loop = []
for i in range(hist_l.n_iters):
    rmse_i = np.sqrt(np.mean((cost_loop[i] - cost_nash)**2))
    rmse_cost_loop.append(rmse_i)
rmse_cost_loop = np.array(rmse_cost_loop)

# %% #################################
# Plots
######################################

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams.update({
    # "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 19,
    "font.size": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 2.5,
})

# Plot the RMSE of the costs over the learning loop
fig_cost, ax_cost = plt.subplots(figsize=(8, 5))
ax_cost.plot(range(hist_l.n_iters), rmse_cost_loop / (max(cost_nash) - min(cost_nash)))
ax_cost.set_xlabel("Iteration of Algorithm 1")
ax_cost.set_ylabel("Normalized RMSE of cost")
ax_cost.set_yscale('log')  # Set y-axis to log scale
# Red cross for iterations where the the closed-loop system unstable (spectral radius > 1)
# label_added = False
# for i in range(hist_l.n_iters):
#     if rad_loop[i] > 1.0:
#         ax_cost.plot(i, rmse_cost_loop[i] / (max(cost_nash) - min(cost_nash)), marker='x', color='red', markersize=3,
#                      label="Unstable iter" if not label_added else "", lw=0)
#         if not label_added:
#             label_added = True
# ax_cost.legend()
ax_cost.grid()
plt.show()
if save_plots:
    fig_cost.savefig(f"figs/fig_lqr_rmse_cost_N_{N}_m_{nx}_k_{n_iters_AL}.eps", format='eps', bbox_inches="tight")

# Plot a closed-loop trajectory using each of the controllers
x_cl_test = x0_test[:, 0]  # Use first test initial condition
T_sim_test = 20
cost_cen_test, X_cen, U_cen, Y_cen = cl_loop(x_cl_test, K_cen, T_sim_test)
cost_pref_test, X_pref, U_pref, Y_pref = cl_loop(x_cl_test, K_pref, T_sim_test)
cost_nash_test, X_nash, U_nash, Y_nash = cl_loop(x_cl_test, K_Nash, T_sim_test)
fig_traj, ax_traj = plt.subplots(figsize=(8, 5))
ax_traj.plot(range(T_sim_test + 1), Y_nash, label="Nash LQR")
ax_traj.plot(range(T_sim_test + 1), Y_pref, '--', label="Preference-based Nash LQR")
ax_traj.plot(range(T_sim_test + 1), Y_cen, '-.', label="Centralized LQR")
ax_traj.set_xlabel("Time step")
ax_traj.set_ylabel("System output")
ax_traj.legend()
ax_traj.grid()
plt.show()
if save_plots:
    fig_traj.savefig(f"figs/fig_lqr_traj_N_{N}_m_{nx}_k_{n_iters_AL}.eps", format='eps', bbox_inches="tight")

# Plot accuracy over the AL loop
fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
ax_acc.plot(hist_l.accuracy, marker='o')
ax_acc.set_xlabel("Iteration of Algorithm 1")
ax_acc.set_ylabel("Accuracy on dataset")
ax_acc.grid()
plt.show()
if save_plots:
    fig_acc.savefig(f"figs/fig_lqr_acc_N_{N}_m_{nx}_k_{n_iters_AL}.eps", format='eps', bbox_inches="tight")

# Plot best-response deviations of the iterates of the AL loop
fig_br_dev, ax_br_dev = plt.subplots(figsize=(8, 5))
for i in range(N):
    ax_br_dev.plot(br_devs[i], label=f"Agent {i + 1}")
ax_br_dev.set_xlabel("Iteration of Algorithm 1")
ax_br_dev.set_ylabel("Best-response deviation")
ax_br_dev.set_yscale('log')
ax_br_dev.legend()
ax_br_dev.grid()
plt.show()
if save_plots:
    fig_br_dev.savefig(f"figs/fig_lqr_br_dev_N_{N}_m_{nx}_k_{n_iters_AL}.eps", format='eps', bbox_inches="tight")

# Some final information about the results
print(f"Normalized RMSE for learned K: {rmse_cost_pref / (max(cost_nash) - min(cost_nash)):0.5f}")
print(f"Maximum BR deviation for learned K: {hist_l.tol_eval[-1]:0.4f}")
