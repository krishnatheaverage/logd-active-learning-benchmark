"""E2 (Reviewer 1.2): ligand-batch acquisition, the realistic decision unit.

In the lab you choose a LIGAND to synthesise and then measure it against many
lanthanides / acidities / diluents. So the acquisition unit here is a ligand:
selecting it reveals ALL its measurements at once. The test set is a disjoint
set of held-out LIGANDS (no leakage). We track, vs. number of ligands acquired:
  * predictive R2 on held-out ligands (does the goal of a good model favour explore?)
  * top-k LIGAND recall (does the goal of finding extractants favour exploit?)
This tests whether the objective-dependent dichotomy survives a chemically
realistic, batched acquisition unit (Reviewer 1.2 + 1.3).
"""
import os, json, numpy as np
import sys; sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import RFUQ, top_k_recall
import data_ext as de

OUT = "results/revision/e2_ligand_batch.json"


def ligand_true_score(y, ligand, ids, agg="max"):
    f = {"max": np.max, "mean": np.mean}[agg]
    return np.array([f(y[ligand == i]) for i in ids])


def run(strategy, Xp, yp, lig_p, Xt, yt, seed, n_seed=6, batch=4, budget_lig=None):
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    ids = np.unique(lig_p)
    budget_lig = budget_lig or int(0.85 * len(ids))
    true_lig_score = ligand_true_score(yp, lig_p, ids, "max")
    true_top = set(np.argsort(-true_lig_score)[:10].tolist())  # indices into ids
    # precompute per-ligand row indices once
    lig_rows = [np.where(lig_p == ids[j])[0] for j in range(len(ids))]
    revealed = set(rng.choice(len(ids), n_seed, replace=False).tolist())
    curve = dict(n_lig=[], n_meas=[], r2=[], recall=[])
    while len(revealed) < budget_lig:
        tr_mask = np.zeros(len(lig_p), bool)
        for j in revealed:
            tr_mask[lig_rows[j]] = True
        model = RFUQ(n=150, seed=seed).fit(Xp[tr_mask], yp[tr_mask])
        mu_t, sd_t = model.predict(Xt)
        r2 = float(1 - np.sum((yt - mu_t) ** 2) / np.sum((yt - yt.mean()) ** 2))
        mu_all, sd_all = model.predict(Xp)  # one batched predict over all pool rows
        best_rev = yp[tr_mask].max()
        unrev = [j for j in range(len(ids)) if j not in revealed]
        exploit, explore = {}, {}
        for j in unrev:
            r = lig_rows[j]; mu = mu_all[r]; sd = sd_all[r]
            if strategy == "random":
                exploit[j] = rng.random()
            elif strategy == "greedy":
                exploit[j] = mu.max()
            elif strategy == "uncertainty":
                exploit[j] = sd.mean()
            elif strategy == "ucb":
                exploit[j] = (mu + 1.5 * sd).max()
            elif strategy == "ei":
                s = np.maximum(sd, 1e-9); z = (mu - best_rev) / s
                exploit[j] = float(((mu - best_rev) * norm.cdf(z) + s * norm.pdf(z)).max())
            else:  # hybrid
                exploit[j] = mu.max(); explore[j] = sd.mean()
        if strategy == "hybrid":
            ex = sorted(unrev, key=lambda j: -exploit[j])[:max(1, batch // 2)]
            rest = [j for j in unrev if j not in set(ex)]
            ep = sorted(rest, key=lambda j: -explore[j])[:batch - len(ex)]
            pick = ex + ep
        else:
            pick = sorted(unrev, key=lambda j: -exploit[j])[:batch]
        revealed |= set(pick)
        curve["n_lig"].append(len(revealed))
        curve["n_meas"].append(int(tr_mask.sum() + sum(len(lig_rows[j]) for j in pick)))
        curve["r2"].append(r2)
        curve["recall"].append(len(revealed & true_top) / len(true_top))
    return curve


def main():
    d = de.load("pool")
    X, y, lig = d["X"], d["y"], d["ligand"]
    ids_all = np.unique(lig)
    strategies = ("random", "greedy", "uncertainty", "ucb", "ei", "hybrid")
    seeds = range(6)
    res = {s: [] for s in strategies}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        test_ids = set(rng.choice(ids_all, int(0.2 * len(ids_all)), replace=False).tolist())
        te = np.isin(lig, list(test_ids)); pool = ~te
        Xp, yp, lig_p = X[pool], y[pool], lig[pool]
        Xt, yt = X[te], y[te]
        for s in strategies:
            res[s].append(run(s, Xp, yp, lig_p, Xt, yt, seed))
    out = {}
    for s in strategies:
        # align on shortest curve length across seeds
        L = min(len(r["r2"]) for r in res[s])
        r2 = np.array([r["r2"][:L] for r in res[s]])
        rec = np.array([r["recall"][:L] for r in res[s]])
        out[s] = dict(n_lig=res[s][0]["n_lig"][:L], n_meas=res[s][0]["n_meas"][:L],
                      r2_mean=r2.mean(0).tolist(), r2_std=r2.std(0).tolist(),
                      recall_mean=rec.mean(0).tolist(), recall_std=rec.std(0).tolist())
        print(f"{s:12s} final R2 {r2.mean(0)[-1]:.3f}  final top10-ligand-recall "
              f"{rec.mean(0)[-1]:.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(unit="ligand", n_ligands=int(len(ids_all)), results=out),
              open(OUT, "w"), indent=2)
    print("E2_DONE", flush=True)


if __name__ == "__main__":
    main()
