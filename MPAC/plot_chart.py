import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============== CONFIG ==============
# Point these to your files (no .pkl needed; the loader tries both)
ALG2FILE = {
    "MPO":  "/home/stav.karasik/MPAC/logs/EVAL__mpo3",
    "CRPO": "/home/stav.karasik/MPAC/logs/EVAL__crpo3",
    "RAMU": "/home/stav.karasik/MPAC/logs/EVAL__ramu3",
    "MPAC": "/home/stav.karasik/MPAC/logs/EVAL__mpac3",
}
SAVE_DIR = "results3"
SAVE_NAME = "rank_distributions.png"
# ====================================

def load_eval(path):
    """Load an eval pickle saved by your evaluation script; accepts files with or without .pkl."""
    if not os.path.exists(path):
        if os.path.exists(path + ".pkl"):
            path = path + ".pkl"
        elif path.endswith(".pkl") and os.path.exists(path[:-4]):
            path = path[:-4]
        else:
            raise FileNotFoundError(f"No such file: {path} or {path}.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    ev = d["eval"]
    # Shapes from your eval script:
    # J_tot: (n_runs, n_perturb)
    # Jc_tot: (n_runs, n_perturb)
    return np.array(ev["J_tot"]), np.array(ev["Jc_tot"]), np.array(ev["perturb_param_values"])

def summarize_per_alg(files_dict):
    """Return dictionaries of per-perturb means for reward and cost, plus shared x (perturb grid)."""
    rewards_mean = {}
    costs_mean = {}
    x_ref = None

    for alg, path in files_dict.items():
        J_tot, Jc_tot, pvals = load_eval(path)
        if x_ref is None:
            x_ref = pvals
        else:
            if not np.allclose(x_ref, pvals):
                raise ValueError(f"Perturbation grid mismatch for {alg}")
        rewards_mean[alg] = J_tot.mean(axis=0)   # mean across runs
        costs_mean[alg]   = Jc_tot.mean(axis=0)  # mean across runs

    return rewards_mean, costs_mean, x_ref

def compute_rank_distribution(means_by_alg, higher_is_better=True):
    """
    For each perturbation value, rank algorithms.
    Return a matrix counts[alg, rank] with occurrences across perturb grid.
    Ranks are 1..K (1 = best).
    """
    algs = list(means_by_alg.keys())
    K = len(algs)
    n_vals = len(next(iter(means_by_alg.values())))
    counts = {alg: np.zeros(K, dtype=int) for alg in algs}  # index r-1 holds count for rank r

    # Build a 2D array shape (K, n_vals) ordered by algs list
    M = np.vstack([means_by_alg[alg] for alg in algs])  # rows=algs, cols=perturb values

    for j in range(n_vals):
        col = M[:, j]
        # Sort to get ranks: best -> worst
        order = np.argsort(-col) if higher_is_better else np.argsort(col)
        # order[0] is best (rank 1), order[1] rank 2, ...
        ranks = np.empty(K, dtype=int)
        ranks[order] = np.arange(1, K + 1)
        for i, alg in enumerate(algs):
            counts[alg][ranks[i]-1] += 1

    # Convert to percentage per rank (stacked bars per rank sum to 100%)
    # For each rank r, total occurrences across algs equals n_vals
    pct = {alg: (counts[alg] / n_vals) * 100.0 for alg in algs}
    return algs, np.vstack([pct[alg] for alg in algs])  # shape (K, K): rows=alg, cols=rank

def stacked_rank_plot(ax, algs, pct_matrix, title):
    """
    ax: matplotlib Axes
    algs: list of algorithm names length K
    pct_matrix: shape (K, K) where rows=alg, cols=rank (1..K) in percent
    """
    K = len(algs)
    x = np.arange(1, K + 1)  # ranks on x-axis
    bottoms = np.zeros(K)
    # stack in the order algs appear; legend will match
    for i, alg in enumerate(algs):
        ax.bar(x, pct_matrix[i], bottom=bottoms, label=alg)
        bottoms += pct_matrix[i]
    ax.set_xticks(x)
    ax.set_xlabel("Rank (1 = best)")
    ax.set_ylabel("Fraction (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

def main():
    rewards_mean, costs_mean, pvals = summarize_per_alg(ALG2FILE)

    # Rank distributions:
    # Rewards: higher is better
    algs_R, pct_R = compute_rank_distribution(rewards_mean, higher_is_better=True)
    # Costs: lower is better
    algs_C, pct_C = compute_rank_distribution(costs_mean, higher_is_better=False)

    # Plot side-by-side stacked bars
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    stacked_rank_plot(ax1, algs_R, pct_R, "Rewards")

    ax2 = plt.subplot(1, 2, 2)
    stacked_rank_plot(ax2, algs_C, pct_C, "Costs")

    # one legend for both
    handles, labels = ax1.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), title=None, bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, SAVE_NAME)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved rank distribution figure to: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
