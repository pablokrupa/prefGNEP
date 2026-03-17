"""
Script to solve Example 1 from:
    F. Salehisadaghiani, W. Shi, and L. Pavel, “An ADMM approach to the problem of
    distributed Nash equilibrium seeking,” CoRR, 2017.
Article can be found at arXiv:1707.01965v2.

In particular, we reproduce here the results from Section VI.B of:
    F. Fabiani and A. Bemporad, “An active learning method for solving competitive
    multiagent decision-making and control problems,” IEEE Transactions on Automatic
    Control, vol. 70, no. 4, pp. 2374–2389, 2024.

(c) 2026 Pablo Krupa
"""

import numpy as np
import jax
import jax.numpy as jnp
from nashopt import GNEP
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
sys.path.append('../src')
from pref_gnep import PrefGNEP  # noqa: E402
from dataset import DataSet  # noqa: E402
from models import gen_quad_models  # noqa: E402

seed = 1234
np.random.seed(seed)

# %% #################################
# Problem setup
######################################

gen_dataset = True  # Whether to generate a new dataset or load existing one
save_plots = True  # Whether to save the plots
ds_size_init = 50  # Dimension of initial dataset
n_iters_AL = 150  # Number of iterations of the AL loop
delta = 0.3  # Exploration parameter in the AL loop
p_delta = 5.0  # Exponent for exploration parameter decay in the AL loop
delta_min = 1e-3  # Minimum value of exploration parameter in the AL loop
sigma = 0.3  # Noise parameter for the AL loop
p_sigma = 4.0  # Exponent for noise parameter decay in the AL loop
sigma_min = 1e-3  # Minimum value of noise parameter in the AL loop

N = 10  # Number of agents
sizes = [1] * N
dim = sum(sizes)  # total dimension


# Agent objectives:
f_real = []
for i in range(N):
    @jax.jit
    def fi(x, p=0.0, i=i):
        val = N * (1. + i / 2.) * x[i] - x[i] * (60.0 * N - jnp.sum(x))
        return val
    f_real.append(fi)

Aeq = None  # No equality constraints
beq = None
lb = 7.0 * np.ones(dim)  # Lower bounds
ub = 100 * np.ones(dim)  # Upper bounds

# Solve real GNEP to get true solution
gnep = GNEP(sizes, f=f_real, lb=lb, ub=ub)

# %% #################################
# Solve problem using preferences
######################################

# Initialize PrefGNEP
th_0, fc, th_min, th_max = gen_quad_models(sizes, full=False, with_linear=True)
pref_gnep = PrefGNEP(sizes=sizes, fc=fc, lb=lb, ub=ub)
pref_gnep.init(th=th_0, th_min=th_min, th_max=th_max)
pref_gnep.loss(rho_th=0.001)
pref_gnep.optimization(adam_epochs=500, adam_eta=0.001)

# Generate initial dataset
if gen_dataset:
    print("\n=== Generating initial dataset and feasibility check ===")
    ds = pref_gnep.generate_initial_dataset(f_real, n_samples=ds_size_init, delta=0.1)
    with open("example_Pavel_Ex1_dataset.pkl", "wb") as temp_file:
        pickle.dump((ds.samples, ds.xA, ds.xB, ds.prefs), temp_file)
else:
    print("\n=== Loading initial dataset and feasibility check ===")
    with open("example_Pavel_Ex1_dataset.pkl", "rb") as temp_file:
        samples, xA, xB, prefs = pickle.load(temp_file)
        ds = DataSet(samples, xA, xB, prefs)

# Test that the dataset is feasible
samples_feasible = pref_gnep.check_sample_feasibility(ds.samples, assert_on_fail=True)
dataset_feasible = pref_gnep.check_dataset_feasibility(ds, assert_on_fail=True)
print(f"All samples feasible: {samples_feasible}")
print(f"All dataset feasible: {dataset_feasible}")

# Initial fit
print("\n=== Fit PrefGNEP using AL method ===")
th_fit, infos, stats = pref_gnep.fit(ds)
pred_th_fit = pref_gnep.predict_dataset(ds)
print(f"Accuracy with init dataset: {pref_gnep.accuracy_score(ds):.4f}")


# Use best-response deviations as evaluation function for the AL loop
def x_star_eval(x_star):
    eval = []
    for i in range(N):
        sol_br = gnep.best_response(i, x_star)
        eval.append(jnp.linalg.norm(sol_br.x - x_star))
    return jnp.array(eval)


# Lear the Nash equilibrium using the active learning loop
x0 = jnp.zeros(dim)
ds, hist_l = pref_gnep.fit_AL_loop(f_real, ds, n_iters=n_iters_AL, x0=x0,
                                   sigma=sigma, p_sigma=p_sigma, sigma_min=sigma_min,
                                   delta=delta, p_delta=p_delta, delta_min=delta_min,
                                   store_gnep_sol=True, store_accuracy=True, f_eval=x_star_eval,
                                   verbose=2, seed=seed)

# Compute the Nash equilibrium with the learned parameters
sol_pref = pref_gnep.solve_gnep(x0)

# %% #################################
# Check and compare results
######################################

sol_gnep = gnep.solve(sol_pref.x, verbose=0)  # True GNE solution
x_star = sol_gnep.x
print("\n=== GNE solution ===")
print(f"x = {np.array2string(x_star, precision=4)}")
print(f"KKT residual norm = {float(jnp.linalg.norm(sol_gnep.res)): 10.7g}")
print(f"KKT evaluations     = {int(sol_gnep.stats.kkt_evals): 3d}")

# Check best responses of all agents at the x_star
print("\n=== Best responses deviation || x_br - x_star || ===")
for i in range(gnep.N):
    sol_br = gnep.best_response(i, x_star)
    print(f"\tAgent {i}: {np.linalg.norm(sol_br.x - x_star):.8f}")

# Check best responses of all agents at the learned GNE
print("\n=== Best responses deviation || x_br - x_pref || ===")
for i in range(gnep.N):
    sol_br = gnep.best_response(i, sol_pref.x)
    print(f"\tAgent {i}: {np.linalg.norm(sol_br.x - sol_pref.x):.8f}")

print(f"\nAccuracy after learning loop: {pref_gnep.accuracy_score(ds):.4f}")
print(f"|| x_star - x_pref ||_inf: {np.array2string(np.linalg.norm(x_star - sol_pref.x, ord=np.inf), precision=4)}")

# %% #################################
# Plots
######################################

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams.update({
    "axes.labelsize": 19,
    "font.size": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 2.5,
})

# Plot hist_l.x_star, compared with the true x_star
x_stars = jnp.array(hist_l.x_star)
fig_x_star, ax_x_star = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10.colors
for i in range(dim):
    ax_x_star.plot(x_stars[:, i], color=colors[i % len(colors)])
    ax_x_star.hlines(x_star[i], 0, hist_l.n_iters, colors=colors[i % len(colors)], linestyles='dashed', lw=2.0)
ax_x_star.set_xlabel("Iteration of Algorithm 1")
ax_x_star.set_ylabel("$x^k$")
ax_x_star.grid()
plt.show()
if save_plots:
    fig_x_star.savefig('figs/fig_Pavel_Ex1_x_star.eps', format='eps', bbox_inches="tight")

# Plot best-response deviations over the learning loop
br_devs = []
for i in range(N):
    br_devs.append([])
for eval in hist_l.eval:
    for i in range(N):
        br_devs[i].append(eval[i])
for i in range(N):
    br_devs[i] = np.array(br_devs[i])

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
    fig_br_dev.savefig('figs/fig_Pavel_Ex1_dev.eps', format='eps', bbox_inches="tight")

# Plot accuracy over the learning loop
fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
ax_acc.plot(hist_l.accuracy, marker='o')
ax_acc.set_xlabel("Iteration of Algorithm 1")
ax_acc.set_ylabel("Accuracy on dataset")
ax_acc.grid()
plt.show()
if save_plots:
    fig_acc.savefig('figs/fig_Pavel_Ex1_acc.eps', format='eps', bbox_inches="tight")
