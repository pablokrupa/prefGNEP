"""
Script to solve Example A.3 from:
    F. Facchinei and C. Kanzow, “Penalty methods for the solution of generalized Nash equilibrium
    problems (with complete test problems),” Sapienza University of Rome, p. 15, 2009.

This example is also solved in Section VI.B of:
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
delta = 0.2  # Exploration parameter in the AL loop
p_delta = 5.0  # Exponent for exploration parameter decay in the AL loop
delta_min = 1e-3  # Minimum value of exploration parameter in the AL loop
sigma = 0.3  # Noise parameter for the AL loop
p_sigma = 4.0  # Exponent for noise parameter decay in the AL loop
sigma_min = 1e-3  # Minimum value of noise parameter in the AL loop

sizes = [3, 2, 2]
dim = sum(sizes)  # total dimension
N = len(sizes)  # number of players

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
i = 0
for size in sizes:
    idx_i_list = list(range(current_index, current_index + size))
    idx_i.append(idx_i_list)
    minus_i_indices = list(range(0, current_index)) + list(range(current_index + size, sum(sizes)))
    idx_minus_i.append(minus_i_indices)
    current_index += size
    idx_i[i] = jnp.array(idx_i[i])
    idx_minus_i[i] = jnp.array(idx_minus_i[i])
    i += 1

# Agent objectives:
f_real = []
for i in range(N):
    @jax.jit
    def fi(x, p=0.0, i=i):
        val = 0.5 * x[idx_i[i]].T @ A[i] @ x[idx_i[i]] + x[idx_i[i]].T @ (B[i] @ x[idx_minus_i[i]] + b[i])
        return jnp.reshape(val, ())
    f_real.append(fi)

Aeq = None
beq = None
lb = -10.0 * jnp.ones(dim)  # Lower bounds
ub = 10.0 * jnp.ones(dim)  # Upper bounds

A_ineq = jnp.array([[1, 1, 1, 0, 0, 0, 0],
                   [1, 1, -1, -1, 0, 0, 1],
                   [0, -1, -1, 1, -1, 1, 0],
                   [-1, 0, -1, 1, 0, 0, 1]])
b_ineq = jnp.array([20.0, 5.0, 7.0, 4.0])
ng = A_ineq.shape[0]


def g(x, p=0.0):
    return A_ineq @ x - b_ineq


# Initialize GNEP solver for later use
gnep = GNEP(sizes, f=f_real, g=g, ng=ng, lb=lb, ub=ub, Aeq=Aeq, beq=beq)

# %% #################################
# Solve problem using preferences
######################################

# Initialize PrefGNEP
th_0, fc, th_min, th_max = gen_quad_models(sizes, full=False, diagonal=False, with_linear=True)
pref_gnep = PrefGNEP(sizes=sizes, fc=fc, g=g, ng=ng, lb=lb, ub=ub, Aeq=Aeq, beq=beq)
pref_gnep.init(th=th_0, th_min=th_min, th_max=th_max)
pref_gnep.loss(rho_th=0.001)
pref_gnep.optimization(adam_epochs=500, adam_eta=0.001)

# Generate initial dataset
if gen_dataset:
    print("\n=== Generating initial dataset and feasibility check ===")
    ds = pref_gnep.generate_initial_dataset(f_real, n_samples=ds_size_init, delta=0.1)
    with open("example_Facchinei_A3_dataset.pkl", "wb") as temp_file:
        pickle.dump((ds.samples, ds.xA, ds.xB, ds.prefs), temp_file)
else:
    print("\n=== Loading initial dataset and feasibility check ===")
    with open("example_Facchinei_A3_dataset.pkl", "rb") as temp_file:
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
print("\n=== Best responses deviation || x_br - x_star ||_inf ===")
for i in range(gnep.N):
    sol_br = gnep.best_response(i, x_star)
    print(f"\tAgent {i}: {np.linalg.norm(sol_br.x - x_star, ord=np.inf):.8f}")

# Check best responses of all agents at the learned GNE
print("\n=== Best responses deviation || x_br - x_pref ||_inf ===")
for i in range(gnep.N):
    sol_br = gnep.best_response(i, sol_pref.x)
    print(f"\tAgent {i}: {np.linalg.norm(sol_br.x - sol_pref.x, ord=np.inf):.8f}")

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
    ax_x_star.hlines(x_star[i], 0, hist_l.n_iters, colors=colors[i % len(colors)], linestyles='dashed', lw=1.3)
ax_x_star.set_xlabel("Iteration of Algorithm 1")
ax_x_star.set_ylabel("$x^k$")
ax_x_star.grid()

# Add zoomed inset for the last 10 iterations
ax_inset = inset_axes(ax_x_star, width="30%", height="30%", loc="upper right")
n_zoom = 10
zoom_start = max(0, len(x_stars) - n_zoom)
for i in range(dim):
    ax_inset.plot(range(zoom_start, len(x_stars)), x_stars[zoom_start:, i], color=colors[i % len(colors)])
    ax_inset.hlines(x_star[i], zoom_start, len(x_stars), colors=colors[i % len(colors)], linestyles='dashed', lw=2.0)
ax_inset.set_xlim(zoom_start, len(x_stars))
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.grid()
mark_inset(ax_x_star, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

plt.show()
if save_plots:
    fig_x_star.savefig('figs/fig_Facchinei_A3_x_star.eps', format='eps', bbox_inches="tight")

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
    fig_br_dev.savefig('figs/fig_Facchinei_A3_br_dev.eps', format='eps', bbox_inches="tight")

# Plot accuracy over the learning loop
fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
ax_acc.plot(hist_l.accuracy, marker='o')
ax_acc.set_xlabel("Iteration of Algorithm 1")
ax_acc.set_ylabel("Accuracy on dataset")
ax_acc.grid()
plt.show()
if save_plots:
    fig_acc.savefig('figs/fig_Facchinei_A3_acc.eps', format='eps', bbox_inches="tight")
