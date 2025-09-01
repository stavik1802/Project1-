import os
import pickle
import numpy as np
import pandas as pd

# ==============================
# 1) Point to your eval outputs
# ==============================
ALG2FILE = {
    "MPO":  "/home/stav.karasik/MPAC/logs/EVAL__mpo3",
    "CRPO": "/home/stav.karasik/MPAC/logs/EVAL__crpo3",
    "RAMU": "/home/stav.karasik/MPAC/logs/EVAL__ramu3",
    "MPAC": "/home/stav.karasik/MPAC/logs/EVAL__mpac3",
}

RESULTS_DIR = "results3"
CSV_PATH = os.path.join(RESULTS_DIR, "summary_normalized.csv")
TEX_PATH = os.path.join(RESULTS_DIR, "summary_normalized.tex")

# ==============================
# 2) Loader that accepts .pkl or no extension
# ==============================
def load_eval(path):
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
    # Arrays: shape (n_runs, n_perturb_values)
    J_tot  = np.asarray(ev["J_tot"])
    Jc_tot = np.asarray(ev["Jc_tot"])
    pvals  = np.asarray(ev["perturb_param_values"])
    return J_tot, Jc_tot, pvals

# ==============================
# 3) Compute per-perturb means
# ==============================
def per_perturb_means(J_tot, Jc_tot):
    # mean across runs for each perturbation value
    reward_mean = J_tot.mean(axis=0)
    cost_mean   = Jc_tot.mean(axis=0)
    return reward_mean, cost_mean

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load all
    rewards = {}
    costs   = {}
    p_ref   = None
    for alg, path in ALG2FILE.items():
        J, Jc, p = load_eval(path)
        r_mu, c_mu = per_perturb_means(J, Jc)
        if p_ref is None:
            p_ref = p
        else:
            if not np.allclose(p_ref, p):
                raise ValueError(f"Perturbation grid mismatch for {alg}")
        rewards[alg] = r_mu
        costs[alg]   = c_mu

    # Sanity: need MPO as baseline
    if "MPO" not in rewards:
        raise KeyError("MPO baseline missing from ALG2FILE.")

    # ==============================
    # 4) Normalize vs MPO per perturbation, then average
    # ==============================
    mpo_reward = rewards["MPO"]
    mpo_cost   = costs["MPO"]

    # Avoid divide-by-zero (shouldn’t happen, but guard anyway)
    eps = 1e-12
    mpo_reward = np.maximum(mpo_reward, eps)
    mpo_cost   = np.maximum(mpo_cost, eps)

    rows = []
    for alg in ALG2FILE.keys():
        r = rewards[alg]
        c = costs[alg]

        # Per-perturb normalization
        norm_r_vals = r / mpo_reward
        norm_c_vals = c / mpo_cost

        # Aggregate across perturbations (simple mean)
        norm_r = float(norm_r_vals.mean())
        norm_c = float(norm_c_vals.mean())

        rows.append({
            "Algorithm": alg,
            "Normalized Reward (vs MPO)": norm_r,
            "Normalized Cost (vs MPO)":   norm_c
        })

    # Sort by Algorithm name (or customize)
    df = pd.DataFrame(rows).set_index("Algorithm").loc[list(ALG2FILE.keys())]

    # Round for presentation
    df_rounded = df.copy()
    df_rounded["Normalized Reward (vs MPO)"] = df_rounded["Normalized Reward (vs MPO)"].map(lambda x: f"{x:.2f}")
    df_rounded["Normalized Cost (vs MPO)"]   = df_rounded["Normalized Cost (vs MPO)"].map(lambda x: f"{x:.2f}")

    # Save CSV (with numeric values, not strings)
    df.to_csv(CSV_PATH, index=True)
    print(f"✅ Saved CSV: {CSV_PATH}")

    # Save LaTeX table
    tex = df_rounded.to_latex(escape=True, column_format="lcc",
                              caption="Performance summary (normalized to MPO) across perturbations.",
                              label="tab:perf_norm_mpo")
    with open(TEX_PATH, "w") as f:
        f.write(tex)
    print(f"✅ Saved LaTeX: {TEX_PATH}")

    # Print a nice console view
    print("\n=== Normalized to MPO (per-perturb normalized, then averaged) ===")
    print(df_rounded)

if __name__ == "__main__":
    main()
