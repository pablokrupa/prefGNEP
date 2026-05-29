"""
Standalone script to load and plot the results of a sensitivity analysis
produced by ``sensitivity_test.py``.

Can also be imported as a module and called programmatically via
``plot_sensitivity_results()``.

Usage (standalone)
------------------
Edit the USER SETTINGS block below and run the script directly::

    python plot_sensitivity.py

(c) 2026 Pablo Krupa
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from problems import get_GNEP  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrors sensitivity_test.py)
# ---------------------------------------------------------------------------

def get_key(v):
    """Canonical float key for dict indexing."""
    return round(v, 8)


def get_trial_data(results, delta, sigma):
    """Return list of trial-metric dicts for a given (delta, sigma) pair."""
    return list(results['data'].get(get_key(delta), {}).get(get_key(sigma), {}).values())


# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------

def plot_sensitivity_results(results, delta_range, sigma_range,
                             br_tol, dist_tol,
                             br_dev_key="br_dev_final_mean",
                             dist_key="dist_final_mean",
                             save_plots=True):
    """Generate and display all sensitivity analysis plots.

    Parameters
    ----------
    results : dict
        Results dict as saved by ``sensitivity_test.py``.
    delta_range : list
        Ordered list of delta values that were tested.
    sigma_range : list
        Ordered list of sigma values that were tested.
    br_tol : float
        Tolerance for the BR-deviation metric (reference line on violin plot).
    dist_tol : float
        Tolerance for the distance-to-GNE metric (reference line on violin plot).
    br_dev_key : str
        Which br_dev metric to use. One of: ``'br_dev_last'``, ``'br_dev_final_mean'``,
        ``'br_dev_final_min'``, ``'br_dev_final_avrg'``.
    dist_key : str
        Which dist metric to use. One of: ``'dist_last'``, ``'dist_final_mean'``,
        ``'dist_final_min'``, ``'dist_final_avrg'``.
    save_plots : bool
        If True, save figures to the ``figs/`` directory.
    """
    problem_name = results.get('info', {}).get('problem_name', 'unknown')
    positions = list(range(1, len(delta_range) + 1))
    tick_labels = [f"{d:.1f}" for d in delta_range]

    # Collect per-delta data for both metrics
    br_data = []
    dist_data = []
    solved_each_delta = []
    for delta in delta_range:
        all_trials = []
        for sigma in sigma_range:
            all_trials.extend(get_trial_data(results, delta, sigma))
        n_total = len(all_trials)
        n_solved = sum(
            t[br_dev_key] < br_tol and t[dist_key] < dist_tol
            for t in all_trials
        )
        solved_each_delta.append(n_solved / n_total if n_total > 0 else 0.0)
        br_data.append([t[br_dev_key] for t in all_trials])
        dist_data.append([t[dist_key] for t in all_trials])

    # ── Bar chart: fraction of trials solved per delta ─────────────────────
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    ax_bar.bar(delta_range, solved_each_delta, width=0.08,
               color='skyblue', edgecolor='black')
    ax_bar.set_xticks(delta_range)
    ax_bar.set_xticklabels(tick_labels)
    ax_bar.set_xlabel('Exploration parameter ($\delta$)')
    ax_bar.set_ylabel('Fraction of trials solved')
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(axis='y')

    # ── Helper to build a violin plot ──────────────────────────────────────
    def _make_violin(ax, data, tol, ylabel):
        vln_data = [d for d in data if len(d) > 1]
        vln_pos = [p for p, d in zip(positions, data) if len(d) > 1]
        if vln_data:
            parts = ax.violinplot(vln_data, positions=vln_pos,
                                  showmedians=True, showextrema=True)
            parts['cmedians'].set_color('orange')
            parts['cmedians'].set_linewidth(2.0)
            for body in parts['bodies']:
                body.set_facecolor('lightblue')
        for pos, d in zip(positions, data):
            ax.scatter([pos] * len(d), d, color='black', s=18, zorder=3, alpha=0.7)
        ax.axhline(tol, color='red', linestyle='--', lw=1.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Exploration parameter $\delta$')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y')

    # ── Violin plot: BR deviation ───────────────────────────────────────────
    fig_br, ax_br = plt.subplots(figsize=(8, 5))
    _make_violin(ax_br, br_data, br_tol, '$\phi$')

    # ── Violin plot: Distance to GNE ───────────────────────────────────────
    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    _make_violin(ax_dist, dist_data, dist_tol, 'Distance to GNE')

    plt.show()

    if save_plots:
        fig_bar.savefig(f'figs/fig_sensitivity_{problem_name}_solve_rate.eps',
                        format='eps', bbox_inches='tight')
        fig_br.savefig(f'figs/fig_sensitivity_{problem_name}_{br_dev_key}_violin.eps',
                       format='eps', bbox_inches='tight')
        fig_dist.savefig(f'figs/fig_sensitivity_{problem_name}_{dist_key}_violin.eps',
                         format='eps', bbox_inches='tight')


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── USER SETTINGS ──────────────────────────────────────────────────────
    problem_name = "Pavel_Ex1"
    prob = get_GNEP(problem_name)
    file_name = f"sensitivity_{prob.name}.pkl"  # File to load (in results/)
    br_dev_key = "br_dev_final_mean"  # BR metric for violin plot
    dist_key = "dist_final_mean"  # Dist metric for violin plot
    save_plots = True  # Save plots to figs/
    # Tolerances — overridden by stored values if present
    dist_tol = prob.dist_tol
    br_tol = prob.br_tol
    # ───────────────────────────────────────────────────────────────────────

    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # "text.usetex": True,
        # "font.family": "Helvetica",
        "axes.labelsize": 19,
        "font.size": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 2.5,
    })

    with open("./results/" + file_name, "rb") as fh:
        results = pickle.load(fh)

    # Reconstruct delta_range and sigma_range from stored keys
    delta_range = sorted(results['data'].keys())
    sigma_range = (sorted(next(iter(results['data'].values())).keys())
                   if results['data'] else [])

    print(f"Loaded results from '{file_name}'.")
    print(f"  problem:      {results.get('info', {}).get('problem_name', 'unknown')}")
    print(f"  delta_range:  {delta_range}")
    print(f"  sigma_range:  {sigma_range}")
    print(f"  br_tol:       {br_tol}")
    print(f"  dist_tol:     {dist_tol}")

    plot_sensitivity_results(results, delta_range, sigma_range,
                             br_tol=br_tol, dist_tol=dist_tol,
                             br_dev_key=br_dev_key, dist_key=dist_key,
                             save_plots=save_plots)
