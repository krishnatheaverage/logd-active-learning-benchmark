import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/real_al_results.json"))
res = d["results"]
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
col = {"random": "k", "greedy": "C3", "uncertainty": "C0", "ucb": "C4", "ei": "C1", "hybrid": "C2"}
lw = {"hybrid": 2.4}

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.5))
for s, r in res.items():
    nq = np.array(r["n_queried"]); c = col.get(s)
    m, sd = np.array(r["r2_mean"]), np.array(r["r2_std"])
    axs[0].plot(nq, m, "-", color=c, lw=lw.get(s, 1.4), label=s)
    axs[0].fill_between(nq, m-sd, m+sd, color=c, alpha=0.10)
    rm, rs = np.array(r["recall_mean"]), np.array(r["recall_std"])
    axs[1].plot(nq, rm, "-", color=c, lw=lw.get(s, 1.4), label=s)
    axs[1].fill_between(nq, rm-rs, rm+rs, color=c, alpha=0.10)
axs[0].axhline(0.85, ls="--", color="gray", lw=0.8)
axs[0].set_xlabel("# experimental measurements"); axs[0].set_ylabel("test R$^2$ (logD)")
axs[0].set_title("Predictive accuracy vs budget"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("# experimental measurements"); axs[1].set_ylabel("top-10 recall")
axs[1].set_title("Design: best extractants found"); axs[1].legend(fontsize=7)
fig.suptitle(f"Active learning on 1202 real logD measurements (pool={d['pool']}, test={d['n_test']})")
fig.tight_layout()
fig.savefig("figures/real_al.pdf"); fig.savefig("figures/real_al.png", dpi=160)
print("wrote figures/real_al.{pdf,png}")
