"""E1 (Reviewer 1.1): how the generalization estimate depends on the split.

Random row-level CV can place measurements of the SAME ligand / source in both
train and test, inflating R2. We compare four splitting regimes with identical
models (RF, 500 trees), reporting R2/RMSE/MAE mean+-sd across folds:
  random    : plain 5-fold over rows (optimistic; what the field usually does)
  ligand    : GroupKFold, no ligand shared between train and test
  source    : GroupKFold, no publication source shared
  family    : GroupKFold over ECFP-Tanimoto ligand families (scaffold-split proxy)
The prospective R2=0.69 (4 new ligands) is the external anchor these should approach.
"""
import os, json, numpy as np
import sys; sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from common import reg_metrics
import data_ext as de

OUT = "results/revision/e1_splits.json"


def _grouped_folds(groups, n_splits, rng):
    """Shuffled group-disjoint folds: partition the UNIQUE groups into n_splits
    bins at random, so repeats give an honest spread (GroupKFold is fixed)."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    bins = np.array_split(uniq, n_splits)
    for b in bins:
        te = np.isin(groups, b)
        yield np.where(~te)[0], np.where(te)[0]


def cv_eval(X, y, groups, kind, n_splits=5, n_repeats=4, seed=0):
    rows = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        if kind == "random":
            folds = KFold(n_splits=n_splits, shuffle=True,
                          random_state=seed + rep).split(X)
        else:
            folds = _grouped_folds(groups, n_splits, rng)
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2 or len(te) < 2:
                continue
            rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                       n_jobs=-1, random_state=seed).fit(X[tr], y[tr])
            rows.append(reg_metrics(y[te], rf.predict(X[te])))
    agg = {m: [r[m] for r in rows] for m in rows[0]}
    return {m: dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                    n=len(v)) for m, v in agg.items()}


def prospective(seed=0):
    d = de.load("pool"); t = de.load("test_new")
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                               n_jobs=-1, random_state=seed).fit(d["X"], d["y"])
    return reg_metrics(t["y"], rf.predict(t["X"]))


def main():
    d = de.load("pool")
    X, y = d["X"], d["y"]
    fam, nfam = de.ligand_family_clusters(d)
    groups = {"ligand": d["ligand"], "source": d["source"], "family": fam}
    res = {}
    res["random"] = cv_eval(X, y, None, "random")
    print(f"random  R2 {res['random']['r2']['mean']:.3f}+-{res['random']['r2']['sd']:.3f}", flush=True)
    for kind, g in groups.items():
        res[kind] = cv_eval(X, y, g, "grouped")
        print(f"{kind:7s} R2 {res[kind]['r2']['mean']:.3f}+-{res[kind]['r2']['sd']:.3f}"
              f"  RMSE {res[kind]['rmse']['mean']:.3f}", flush=True)
    res["prospective_new_ligands"] = prospective()
    print(f"prospective (4 new ligands) R2 {res['prospective_new_ligands']['r2']:.3f}", flush=True)
    meta = dict(n_rows=int(len(y)), n_ligands=int(len(np.unique(d["ligand"]))),
                n_sources=int(len(np.unique(d["source"]))), n_families=int(nfam))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(meta=meta, results=res), open(OUT, "w"), indent=2)
    print("E1_DONE", flush=True)


if __name__ == "__main__":
    main()
