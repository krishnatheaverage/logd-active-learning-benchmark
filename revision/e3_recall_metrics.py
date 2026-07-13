"""E3 (Reviewer 1.3): what does 'top-k recall' mean, and richer discovery metrics.

The reviewer notes discovery value differs depending on whether a 'top system' is
a unique LIGAND, a ligand-METAL pair, or a condition-specific ROW. We re-report
recall under all three definitions on the original row-level benchmark, and add
enrichment factor, simple regret, and hit-rate above a practical logD threshold
(logD >= 1  <=>  distribution coefficient D >= 10).
"""
import os, json, numpy as np
import sys; sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import RFUQ, enrichment_factor, regret, hit_rate
import data_ext as de

OUT = "results/revision/e3_recall_metrics.json"
K = 10
THRESH = 1.0  # logD >= 1  <=>  D >= 10


def group_recall(revealed, y, key, k=K):
    """Fraction of the true top-k GROUPS (by max y within group) that have at
    least one revealed member. key is a per-row group label array."""
    ids = np.unique(key)
    gmax = np.array([y[key == i].max() for i in ids])
    top_groups = set(ids[np.argsort(-gmax)[:k]].tolist())
    rev = set(np.asarray(revealed).tolist())
    hit = sum(1 for g in top_groups if any((key == g) & _mask(len(y), rev)))
    return hit / len(top_groups)


def _mask(n, idx_set):
    m = np.zeros(n, bool)
    if idx_set:
        m[np.fromiter(idx_set, int)] = True
    return m


def run(strategy, X, y, lig, met, seed, n_seed=25, batch=15, budget=420):
    rng = np.random.default_rng(seed)
    N = len(y)
    pair = np.array([f"{a}|{b}" for a, b in zip(lig, met)])
    revealed = list(rng.choice(N, n_seed, replace=False))
    curve = {kk: [] for kk in ["n", "rec_row", "rec_pair", "rec_lig",
                                "enrich", "regret", "hit"]}
    top_rows = set(np.argsort(-y)[:K].tolist())
    while len(revealed) < budget:
        rmask = np.ones(N, bool); rmask[revealed] = False
        unrev = np.where(rmask)[0]
        model = RFUQ(n=200, seed=seed).fit(X[revealed], y[revealed])
        mu, sd = model.predict(X[unrev]); best = y[revealed].max()
        if strategy == "random":
            score = rng.random(len(unrev))
        elif strategy == "greedy":
            score = mu
        elif strategy == "uncertainty":
            score = sd
        elif strategy == "ucb":
            score = mu + 1.5 * sd
        elif strategy == "ei":
            from scipy.stats import norm
            s = np.maximum(sd, 1e-9); z = (mu - best) / s
            score = (mu - best) * norm.cdf(z) + s * norm.pdf(z)
        elif strategy == "hybrid":
            ne = batch // 2
            ex = list(unrev[np.argsort(-mu)[:ne]])
            rest = [i for i in unrev if i not in set(ex)]
            sr = sd[np.isin(unrev, rest)]
            ep = list(np.array(rest)[np.argsort(-sr)[:batch - ne]])
            pick = np.array(ex + ep); revealed += list(pick)
            _record(curve, revealed, y, lig, met, top_rows, N); continue
        pick = unrev[np.argsort(-score)[:batch]]
        revealed += list(pick)
        _record(curve, revealed, y, lig, met, top_rows, N)
    return curve


def _record(curve, revealed, y, lig, met, top_rows, N):
    pair = np.array([f"{a}|{b}" for a, b in zip(lig, met)])
    curve["n"].append(len(revealed))
    curve["rec_row"].append(len(set(revealed) & top_rows) / K)
    curve["rec_pair"].append(group_recall(revealed, y, pair))
    curve["rec_lig"].append(group_recall(revealed, y, lig))
    curve["enrich"].append(enrichment_factor(y, revealed, k=K))
    curve["regret"].append(regret(y, revealed))
    curve["hit"].append(hit_rate(y, revealed, THRESH))


def main():
    d = de.load("pool")
    X, y, lig, met = d["X"], d["y"], d["ligand"], d["lanthanide"]
    strategies = ("random", "greedy", "uncertainty", "ucb", "ei", "hybrid")
    seeds = range(6)
    res = {}
    for s in strategies:
        runs = [run(s, X, y, lig, met, seed) for seed in seeds]
        L = min(len(r["n"]) for r in runs)
        agg = {}
        for kk in ["rec_row", "rec_pair", "rec_lig", "enrich", "regret", "hit"]:
            A = np.array([r[kk][:L] for r in runs])
            agg[kk + "_mean"] = A.mean(0).tolist()
            agg[kk + "_std"] = A.std(0).tolist()
        agg["n"] = runs[0]["n"][:L]
        res[s] = agg
        print(f"{s:12s} rec_row {agg['rec_row_mean'][-1]:.2f}  rec_pair "
              f"{agg['rec_pair_mean'][-1]:.2f}  rec_lig {agg['rec_lig_mean'][-1]:.2f}  "
              f"enrich {agg['enrich_mean'][-1]:.1f}  hit {agg['hit_mean'][-1]:.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(K=K, threshold=THRESH, results=res), open(OUT, "w"), indent=2)
    print("E3_DONE", flush=True)


if __name__ == "__main__":
    main()
