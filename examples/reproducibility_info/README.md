# Reproducibility Instructions

This readme describes how to recreate the exact software environment used to generate the numerical results in the accompanying paper shown below, and how to re-run the example scripts.

```bibtex
@article{Krupa_prefGNEP_arXiv_2026,
  title = {Learning generalized {N}ash equilibria from pairwise preferences},
  author = {Krupa, Pablo and Bemporad, Alberto},
  year = {2026},
  journal = {arXiv preprint arXiv:xxxx.yyyy},
}
```

---

## Requirements

- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) — version **25.7.0** (recommended, as this is the version used to produce the results)
- Python **3.13.7** virtual environment (installed automatically by the steps below)

---

## Step 1 — Install Miniconda

Download and install Miniconda (recommended version 25.7.0) following instructions for your operating system on the [Miniconda website](https://www.anaconda.com/docs/getting-started/miniconda/install).

After installation, verify the version:

```bash
conda --version
```

---

## Step 2 — Recreate the virtual environment

The files needed to recreate the environment (`spec-file.txt`, `environment.yml`, `requirements.txt`) are located in `examples/reproducibility_info/` directory. From this directory, run the following commands, replacing `<ENV_NAME>` with a name of your choice (e.g., `prefGNEPenv`):

```bash
conda create --name <ENV_NAME> --file spec-file.txt
conda env update --name <ENV_NAME> --file environment.yml
conda activate <ENV_NAME>
python -m pip install -r requirements.txt
```

This will install all packages at the exact versions used to generate the reported results, including Python 3.13.7.

> **Note:** `spec-file.txt` is platform-specific (macOS). If you are on a different operating system, the `conda create` step may fail. In that case, use `environment.yml` alone to create the environment:
> ```bash
> conda env create --name <ENV_NAME> --file environment.yml
> conda activate <ENV_NAME>
> python -m pip install -r requirements.txt
> ```
> Package versions may differ slightly from those used in the paper.

---

## Step 3 — Run the example scripts

With the environment activated, navigate to the `examples/` directory and run any of the `example_*.py` scripts:

```bash
python example_lqr.py
python example_Picheny.py
python example_Facchinei_A3.py
python example_Pavel_Ex1.py
```

To fully reproduce the game-theoretic LQR results, run `example_lqr.py` changing the `problem_size` variable to `"small"`, `"medium"` and `"large"`, and the `n_iters_AL` variable to `100` and `200`.

Each script is self-contained and will reproduce the corresponding figures and numerical results from the paper. Output figures are saved as `.eps` files in `examples/figs/`.
