import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1) Define your evaluation files
# ==============================
ALG2FILE = {
    "MPO":  "/home/stav.karasik/MPAC/logs/EVAL__mpo3",
    "CRPO": "/home/stav.karasik/MPAC/logs/EVAL__crpo3",
    "RAMU": "/home/stav.karasik/MPAC/logs/EVAL__ramu3",
    "MPAC": "/home/stav.karasik/MPAC/logs/EVAL__mpac3",
}

# ==============================
# 2) Loader that accepts .pkl or no extension
# ==============================
def load_eval(path):
    if not os.path.exists(path):
        if os.path.exists(path + ".pkl"):
            path = path + ".pkl"
        elif os.path.exists(path.rstrip(".pkl")):
            path = path.rstrip(".pkl")
        else:
            raise FileNotFoundError(f"No such file: {path} or {path}.pkl")
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)

    eval_d = d["eval"]
    return (
        np.array(eval_d["J_tot"]),
        np.array(eval_d["Jc_tot"]),
        np.array(eval_d["perturb_param_values"]),
    )

# ==============================
# 3) Summarize across runs
# ==============================
def summarize(J_tot, Jc_tot):
    reward_mean = J_tot.mean(axis=0)
    reward_std  = J_tot.std(axis=0)
    cost_mean   = Jc_tot.mean(axis=0)
    cost_std    = Jc_tot.std(axis=0)
    return reward_mean, reward_std, cost_mean, cost_std

# ==============================
# 4) Load and summarize all algorithms
# ==============================
summaries = {}
pert_vals_ref = None

for alg, pth in ALG2FILE.items():
    J_tot, Jc_tot, pvals = load_eval(pth)
    if pert_vals_ref is None:
        pert_vals_ref = pvals
    else:
        if not np.allclose(pert_vals_ref, pvals):
            raise ValueError(f"Perturbation grid mismatch for {alg}.")
    summaries[alg] = summarize(J_tot, Jc_tot)

# ==============================
# 5) Plot
# ==============================
plt.figure(figsize=(12, 5))

# Reward subplot
ax1 = plt.subplot(1, 2, 1)
for alg, (r_mean, r_std, _, _) in summaries.items():
    ax1.plot(pert_vals_ref, r_mean, marker="o", label=alg)
    ax1.fill_between(pert_vals_ref, r_mean - r_std, r_mean + r_std, alpha=0.15)
ax1.set_title("Total Reward vs Perturbation")
ax1.set_xlabel("Perturbation Value")
ax1.set_ylabel("Total Reward")
ax1.grid(True, alpha=0.3)
ax1.legend(title="Algorithm")

# Cost subplot
ax2 = plt.subplot(1, 2, 2)
for alg, (_, _, c_mean, c_std) in summaries.items():
    ax2.plot(pert_vals_ref, c_mean, marker="o", label=alg)
    ax2.fill_between(pert_vals_ref, c_mean - c_std, c_mean + c_std, alpha=0.15)
ax2.set_title("Total Cost vs Perturbation")
ax2.set_xlabel("Perturbation Value")
ax2.set_ylabel("Total Cost")
ax2.grid(True, alpha=0.3)
ax2.legend(title="Algorithm")

plt.tight_layout()

# ==============================
# 6) Save results
# ==============================
results_dir = "results3"
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, "merged_results.png")
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to {save_path}")
plt.close()
