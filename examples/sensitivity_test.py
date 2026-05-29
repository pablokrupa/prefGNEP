"""
This script performs a sensitivity analysis of the AL method to study the effect of choice of
the initial exploration term delta_0.

(c) 2026 Pablo Krupa
"""

import numpy as np
import psutil
import jax
import jax.numpy as jnp
from nashopt import GNEP
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import hashlib
import pickle
import gc
import os
import sys
from datetime import datetime
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.append('../src')
from pref_gnep import PrefGNEP  # noqa: E402
from dataset import DataSet  # noqa: E402
from models import gen_quad_models  # noqa: E402
from problems import get_GNEP  # noqa: E402
from plot_sensitivity import plot_sensitivity_results  # noqa: E402

# Terminal colour codes
CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
ORANGE = "\033[33m"
GRAY = "\033[90m"
RESET = "\033[0m"

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 19,
    "font.size": 20,
    "legend.fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 2.5,
})

seed = 1234
np.random.seed(seed)

# %% #################################
# Problem setup
######################################

save_plots = True  # Whether to save the plots
ds_size_init = 50  # Dimension of initial dataset
n_iters_AL = 150  # Number of iterations of the AL loop
delta_range = [0.1, 0.2, 0.3, 0.4]  # Range of exploration parameters to test
p_delta = 5.0  # Exponent for exploration parameter decay in the AL loop
delta_min = 1e-4  # Minimum value of exploration parameter in the AL loop
sigma_range = [0.3]  # Range of noise parameters to test
p_sigma = 3.0  # Exponent for noise parameter decay in the AL loop
sigma_min = 1e-3  # Minimum value of noise parameter in the AL loop
num_trials = 10  # Number of trials to run for each combination of parameters
num_final = 5  # Number of final iterations to consider when evaluating the best parameters found during the AL loop

problem_name = "Picheny"  # "Picheny" | "Facchinei_A3" | "Pavel_Ex1"

# Load problem definition
prob = get_GNEP(problem_name)
sizes = prob.sizes
dim = sum(sizes)
N = len(sizes)
f_real = prob.f_real
lb, ub = prob.lb, prob.ub
Aeq, beq = prob.Aeq, prob.beq
g, ng = prob.g, prob.ng
h, nh = prob.h, prob.nh
dist_tol = prob.dist_tol
br_tol = prob.br_tol
print(f"Running sensitivity analysis tests for problem: {prob.name}.")
print(f"Distance to GNE tolerance: {dist_tol:.3f}. BR tolerance: {br_tol:.3f}.")

continue_from_previous = False  # Whether to continue from a previous run (if True, results will be loaded from file)
load_file_name = f"sensitivity_{prob.name}.pkl"  # Name of file to load results from

# Solve real GNEP to get true solution
gnep = GNEP(sizes, f=f_real, lb=lb, ub=ub, g=g, ng=ng, Aeq=Aeq, beq=beq, h=h, nh=nh)
x0 = jnp.zeros(dim)
sol_gnep = gnep.solve(x0, verbose=0)  # True GNE solution
x_star = sol_gnep.x

# Generate name for results file
if continue_from_previous:
    print(f"Continuing from previous run. Results will be loaded from file: {load_file_name}")
    name_res_file = load_file_name
else:
    base_name = f"sensitivity_{prob.name}"
    date = datetime.now().strftime("%Y_%m_%d")
    name_res_file = f"{base_name}_{date}.pkl"
    _res_path = "./results/" + name_res_file
    if __import__('os').path.exists(_res_path):
        print(f"\n{ORANGE}{'!' * 60}")
        print(f"  WARNING: Results file already exists: {name_res_file}")
        print("  Starting a new run will OVERWRITE this file and all")
        print("  previously saved results will be lost.")
        print(f"{'!' * 60}\033[0m")
        _confirm = input("  Type 'yes' to continue and overwrite, or anything else to abort: ").strip().lower()
        if _confirm != "yes":
            print("Aborted. Set 'continue_from_previous = True' to resume the existing run.")
            raise SystemExit(0)


fig_dir = "./figs/" + name_res_file.replace(".pkl", "")
log_path = fig_dir + "/log.txt"
os.makedirs(fig_dir, exist_ok=True)

# Write log header for new runs; open in append mode for continuations
if save_plots:
    _log_mode = "a" if continue_from_previous else "w"
    with open(log_path, _log_mode) as _lf:
        if not continue_from_previous:
            _lf.write(f"{'─' * 60}\n")
            _lf.write(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _lf.write(f"Problem:      {prob.name}\n")
            _lf.write(f"Base seed:    {seed}\n")
            _lf.write(f"n_iters_AL:   {n_iters_AL}  |  ds_size_init: {ds_size_init}  |  num_final: {num_final}\n")
            _lf.write(f"delta_range:  {delta_range}\n")
            _lf.write(f"sigma_range:  {sigma_range}\n")
            _lf.write(f"p_delta:      {p_delta}  |  delta_min: {delta_min}\n")
            _lf.write(f"p_sigma:      {p_sigma}  |  sigma_min: {sigma_min}\n")
            _lf.write(f"dist_tol:     {dist_tol}  |  br_tol: {br_tol}\n")
            _lf.write(f"{'─' * 60}\n\n")


def save_results(results, file_name):
    # Helper function to save results to file
    with open("./results/" + file_name, "wb") as temp_file:
        pickle.dump(results, temp_file)
    print(f"\t\tResults saved to file: {file_name}")


# %% #################################
# Initialize PrefGNEP
######################################

# Initialize PrefGNEP
th_0, fc, th_min, th_max = gen_quad_models(prob.sizes, **prob.quad_model_kwargs)
pref_gnep = PrefGNEP(sizes=prob.sizes, fc=fc, lb=prob.lb, ub=prob.ub,
                     g=prob.g, ng=prob.ng, Aeq=prob.Aeq, beq=prob.beq,
                     h=prob.h, nh=prob.nh, clear_jax_cache=True)
pref_gnep.init(th=th_0, th_min=th_min, th_max=th_max)
pref_gnep.loss(rho_th=0.001)
pref_gnep.optimization(adam_epochs=500, adam_eta=0.001)


# Use best-response deviations as evaluation function for the AL loop
def x_star_eval(x_star):
    eval = []
    for i in range(N):
        sol_br = gnep.best_response(i, x_star)
        eval.append(jnp.linalg.norm(sol_br.x - x_star))
    return jnp.array(eval)


def get_trial_seed(base_seed, delta, sigma, trial):
    # Hash-based seed: deterministic and independent of array ordering
    key = f"{base_seed}_{delta:.8f}_{sigma:.8f}_{trial}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31 - 1)


def get_key(v):
    # Canonical float key for dict indexing (avoids floating-point comparison issues)
    return round(v, 8)


def trial_done(results, delta, sigma, trial):
    # Check if a trial has already been completed
    return results['data'].get(get_key(delta), {}).get(get_key(sigma), {}).get(trial) is not None


def get_trial_data(results, delta, sigma):
    # Return list of dicts for all completed trials of a given (delta, sigma) combination
    return list(results['data'].get(get_key(delta), {}).get(get_key(sigma), {}).values())


# Initialize variables to store results
if continue_from_previous:
    print(f"Loading results from file: {load_file_name}")
    with open("./results/" + load_file_name, "rb") as temp_file:
        results = pickle.load(temp_file)
    results['info']['base_seed'] = seed
    results['info']['dist_tol'] = dist_tol
    results['info']['br_tol'] = br_tol
else:
    results = {'data': {}, 'info': {'base_seed': seed, 'dist_tol': dist_tol, 'br_tol': br_tol, 'problem_name': prob.name}}
    # Save initial results to file
    save_results(results, name_res_file)

# %% #################################
# Sensitivity analysis tests
######################################

num_tests = len(delta_range) * len(sigma_range)
curr_test = 0
print(f"Starting sensitivity analysis tests. Numer of delta values: {len(delta_range)}, "
      f"number of sigma values: {len(sigma_range)}, trials per combination: {num_trials}.")


def eval_solution(x_pref_in, x_star_in):
    """Evaluate a candidate solution x_pref_in by computing its distance to the true
    solution x_star and its best-response deviation."""
    eval = []
    for i in range(N):
        sol_br = gnep.best_response(i, x_pref_in)
        eval.append(np.linalg.norm(sol_br.x - x_pref_in) / np.linalg.norm(sol_br.x))
    eval_br = np.max(np.array(eval))
    eval_dist = np.linalg.norm(x_pref_in - x_star_in) / np.linalg.norm(x_star_in)
    return eval_dist, eval_br


def eval_final_min(hist_l, x_star_in, num_final=num_final):
    eval_dist = []
    eval_br = []
    for x_pref_k in hist_l.x_star[-num_final:]:
        eval_dist_k, eval_br_k = eval_solution(x_pref_k, x_star_in)
        eval_dist.append(eval_dist_k)
        eval_br.append(eval_br_k)
    min_eval_dist = np.min(np.array(eval_dist))
    min_eval_br = np.min(np.array(eval_br))
    return min_eval_dist, min_eval_br


def eval_final_avrg(hist_l, x_star_in, num_final=num_final):
    eval_dist = []
    eval_br = []
    for x_pref_k in hist_l.x_star[-num_final:]:
        eval_dist_k, eval_br_k = eval_solution(x_pref_k, x_star_in)
        eval_dist.append(eval_dist_k)
        eval_br.append(eval_br_k)
    avrg_eval_dist = np.mean(np.array(eval_dist), axis=0)
    avrg_eval_br = np.mean(np.array(eval_br), axis=0)
    return avrg_eval_dist, avrg_eval_br


def _fmt_br(val, tol=br_tol):
    color = GREEN if val < tol else RED
    return f"{color}{val:.4f}{RESET}"


def _fmt_dist(val, tol=dist_tol):
    color = GREEN if val < tol else RED
    return f"{color}{val:.4f}{RESET}"


for delta in delta_range:
    for sigma in sigma_range:
        curr_test += 1
        print(f"{'─' * 60}")
        print(f"{CYAN}> Running test {curr_test}/{num_tests} (delta={delta:0.2f}, sigma={sigma:0.2f}){RESET}")

        # Report already-completed trials
        skipped = [t for t in range(num_trials) if trial_done(results, delta, sigma, t)]
        if skipped:
            print(f"{GRAY}\tTrials {skipped[0] + 1}-{skipped[-1] + 1} already done."
                  f" Skipping {len(skipped)} trial(s)...{RESET}")

        for trial in range(num_trials):
            seed_trial = get_trial_seed(seed, delta, sigma, trial)

            # Skip this trial if it has already been completed
            if trial_done(results, delta, sigma, trial):
                continue

            print(f"{BLUE}> Trial {trial + 1}/{num_trials} of test {curr_test}/{num_tests}"
                  f" (with delta={delta:0.2f} and sigma={sigma:0.2f}){RESET}")
            _trial_start = datetime.now()

            # Generate initial dataset
            ds = pref_gnep.generate_initial_dataset(f_real, n_samples=ds_size_init, delta=0.1, seed=seed_trial, verbose=0)

            # Initial fit
            th_fit, infos, stats = pref_gnep.fit(ds, th_0=th_0)

            # Run AL loop
            ds, hist_l = pref_gnep.fit_AL_loop(f_real, ds, n_iters=n_iters_AL, x0=x0,
                                               sigma=sigma, p_sigma=p_sigma, sigma_min=sigma_min,
                                               delta=delta, p_delta=p_delta, delta_min=delta_min,
                                               store_gnep_sol=True, store_accuracy=True, f_eval=x_star_eval,
                                               verbose=2, seed=seed_trial, update_th_0=True)

            # Compute the Nash equilibrium with the learned parameters
            sol_pref = pref_gnep.solve_gnep(x0)

            # Compute the average of the last num_final solutions
            x_pref_mean = np.mean(hist_l.x_star[-5:], axis=0)

            # Recompute the Nash equilibrium using x_pref_mean as initial guess
            sol_gnep = gnep.solve(x_pref_mean, verbose=0)
            x_star = sol_gnep.x

            # Compute best response deviation
            dist_last, br_dev_last = eval_solution(sol_pref.x, x_star)
            dist_final_mean, br_dev_final_mean = eval_solution(x_pref_mean, x_star)
            dist_final_min, br_dev_final_min = eval_final_min(hist_l, x_star)
            dist_final_avrg, br_dev_final_avrg = eval_final_avrg(hist_l, x_star)

            dist_last = np.linalg.norm(sol_pref.x - x_star) / np.linalg.norm(x_star)
            dist_final_mean = np.linalg.norm(x_pref_mean - x_star) / np.linalg.norm(x_star)

            # Store results for this trial
            results['data'].setdefault(get_key(delta), {}).setdefault(get_key(sigma), {})[trial] = {
                'br_dev_last': br_dev_last, 'br_dev_final_mean': br_dev_final_mean,
                'br_dev_final_min': br_dev_final_min, 'br_dev_final_avrg': br_dev_final_avrg,
                'dist_last': dist_last, 'dist_final_mean': dist_final_mean,
                'dist_final_min': dist_final_min, 'dist_final_avrg': dist_final_avrg,
                'acc_last': hist_l.accuracy[-1],
                'time': np.sum(hist_l.times),
            }

            # Some verbose output for monitoring
            print(f"\t\tDist to GNE. \tmean: {_fmt_dist(dist_final_mean)}, last: {_fmt_dist(dist_last)}"
                  f"  final_min: {_fmt_dist(dist_final_min)},  final_avrg: {_fmt_dist(dist_final_avrg)}.")
            print(f"\t\tBR deviation. \tmean: {_fmt_br(br_dev_final_mean)}, last: {_fmt_br(br_dev_last)},"
                  f" final_min: {_fmt_br(br_dev_final_min)}, final_avrg: {_fmt_br(br_dev_final_avrg)}.")

            # Save results after each trial
            save_results(results, name_res_file)

            # Log results (only if saving plots, results are in the pkl file regardless)
            if save_plots:
                with open(log_path, "a") as _lf:
                    _lf.write(f"> delta={delta:.2f}, sigma={sigma:.2f}, trial={trial + 1}\n")
                    _lf.write(
                        f"  br_dev_last: {br_dev_last:.6f}  br_dev_final_mean: {br_dev_final_mean:.6f}"
                        f"  br_dev_final_min: {br_dev_final_min:.6f}  br_dev_final_avrg: {br_dev_final_avrg:.6f}\n"
                    )
                    _lf.write(
                        f"  dist_last:   {dist_last:.6f}  dist_final_mean:   {dist_final_mean:.6f}"
                        f"  dist_final_min:   {dist_final_min:.6f}  dist_final_avrg:   {dist_final_avrg:.6f}\n"
                    )
                    _lf.write(f"  acc_last: {hist_l.accuracy[-1]:.4f}  time: {np.sum(hist_l.times):.1f} s\n\n")

            if save_plots:
                plot_name_base = f"sensitivity_{prob.name}_delta{delta:.2f}_sigma{sigma:.2f}_trial{trial + 1}"

                # Plot hist_l.x_star, compared with the true x_star
                x_stars = jnp.array(hist_l.x_star)
                fig_x_star, ax_x_star = plt.subplots(figsize=(8, 5))
                colors = plt.cm.tab10.colors
                for i in range(dim):
                    ax_x_star.plot(x_stars[:, i], color=colors[i % len(colors)])
                    ax_x_star.hlines(x_star[i], 0, hist_l.n_iters,
                                     colors=colors[i % len(colors)], linestyles='dashed', lw=2.0)
                ax_x_star.set_xlabel("Iteration of Algorithm 1")
                ax_x_star.set_ylabel("$x^k$")
                ax_x_star.grid()
                if save_plots:
                    fig_x_star.savefig(f'{fig_dir}/{plot_name_base}_x_star.eps', format='eps', bbox_inches="tight")
                    plt.close(fig_x_star)

                if prob.name == "Facchinei_A3":
                    # Add zoomed inset for the last 10 iterations
                    ax_inset = inset_axes(ax_x_star, width="30%", height="30%", loc="upper right")
                    n_zoom = 10
                    zoom_start = max(0, len(x_stars) - n_zoom)
                    for i in range(dim):
                        ax_inset.plot(range(zoom_start, len(x_stars)), x_stars[zoom_start:, i],
                                      color=colors[i % len(colors)])
                        ax_inset.hlines(x_star[i], zoom_start, len(x_stars),
                                        colors=colors[i % len(colors)], linestyles='dashed', lw=2.0)
                    ax_inset.set_xlim(zoom_start, len(x_stars))
                    ax_inset.set_xticks([])
                    ax_inset.set_yticks([])
                    ax_inset.grid()
                    mark_inset(ax_x_star, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

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
                # ax_br_dev.legend()
                ax_br_dev.grid()
                if save_plots:
                    fig_br_dev.savefig(f'{fig_dir}/{plot_name_base}_br_dev.eps', format='eps', bbox_inches="tight")
                    plt.close(fig_br_dev)

                # Plot accuracy over the learning loop
                fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
                ax_acc.plot(hist_l.accuracy, marker='o')
                ax_acc.set_xlabel("Iteration of Algorithm 1")
                ax_acc.set_ylabel("Accuracy on dataset")
                ax_acc.grid()
                if save_plots:
                    fig_acc.savefig(f'{fig_dir}/{plot_name_base}_acc.eps', format='eps', bbox_inches="tight")
                    plt.close(fig_acc)

            # Free large per-trial objects to prevent memory accumulation across trials
            del ds, hist_l, sol_pref
            gc.collect()
            jax.clear_caches()

            # Memory usage after cleanup
            # _proc = psutil.Process(os.getpid())
            # _mem = _proc.memory_info()
            # print(f"{GRAY}\t\tTime of trial: {(datetime.now() - _trial_start).total_seconds() / 60:.1f} min."
            #       f" Memory: {_mem.rss / 1e9:.2f} GB{RESET}")

    # Delta summary across all sigma values
    print(f"\n{CYAN}  Delta={delta:.2f} summary:{RESET}")
    for sigma in sigma_range:
        completed = get_trial_data(results, delta, sigma)
        if completed:
            n_done = len(completed)
            n_solved_dist = sum(t['dist_final_mean'] < dist_tol for t in completed)
            n_solved_br = sum(t['br_dev_final_mean'] < br_tol for t in completed)
            med_br_dev = np.median([t['br_dev_final_mean'] for t in completed])
            med_dist = np.median([t['dist_final_mean'] for t in completed])
            print(f"    sigma={sigma:.2f}: {n_solved_dist}/{n_done} solved (dist), {n_solved_br}/{n_done} solved (br)"
                  f" median br_dev_final_mean = {_fmt_br(med_br_dev)} (tol: {br_tol})"
                  f", median dist_final_mean = {_fmt_dist(med_dist)} (tol: {dist_tol})")

# Final print to user
print(f"\nSensitivity analysis tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
      f" Final results saved to file: {name_res_file}")

# %% #################################
# Plots and results analysis
######################################

plot_sensitivity_results(results, delta_range, sigma_range,
                         br_tol=br_tol, dist_tol=dist_tol, save_plots=save_plots)
