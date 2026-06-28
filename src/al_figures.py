import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/al_results.json"))
res = d["results"]
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4))
colors = {"random": "k", "greedy": "C3", "uncertainty": "C0", "ucb": "C2", "ei": "C1"}
for strat, r in res.items():
    nq = np.array(r["n_queried"]); c = colors.get(strat, None)
    bm, bs = np.array(r["best_mean"]), np.array(r["best_std"])
    axs[0].plot(nq, bm, "-", color=c, label=strat)
    axs[0].fill_between(nq, bm-bs, bm+bs, color=c, alpha=0.12)
    rm, rs = np.array(r["recall_mean"]), np.array(r["recall_std"])
    axs[1].plot(nq, rm, "-", color=c, label=strat)
    axs[1].fill_between(nq, rm-rs, rm+rs, color=c, alpha=0.12)
axs[0].set_xlabel("# oracle queries"); axs[0].set_ylabel("best donor strength found")
axs[0].set_title(f"Best-found vs budget (pool={d['n_pool']})"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("# oracle queries"); axs[1].set_ylabel("top-10 recall")
axs[1].set_title("Top-10 recall vs budget"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/al_learning_curves.pdf"); fig.savefig("figures/al_learning_curves.png", dpi=160)
print("wrote figures/al_learning_curves.{pdf,png}")
