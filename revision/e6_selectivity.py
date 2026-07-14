"""E6 (Reviewer 1.5, selectivity clause): a high distribution coefficient alone is
not sufficient for separation. We compute, per ligand, the SEPARATION FACTOR across
the 14 lanthanides (spread of logD = log of the best/worst distribution-coefficient
ratio) and show that ranking ligands by extraction strength (max logD) is NOT the
same as ranking by selectivity, and give the logD-vs-selectivity Pareto front.
"""
import os, json, numpy as np
import sys; sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.stats import spearmanr
import data_ext as de

OUT = "results/revision/e6_selectivity.json"
MIN_METALS = 5  # need enough lanthanides to define a spread


def main():
    d = de.load("pool")
    y, lig, met = d["y"], d["ligand"], d["lanthanide"]
    ids = np.unique(lig)
    rows = []
    for i in ids:
        m = lig == i
        mets = met[m]
        # per-lanthanide typical logD for this ligand (mean over conditions)
        per = {a: y[m][mets == a].mean() for a in np.unique(mets)}
        if len(per) < MIN_METALS:
            continue
        vals = np.array(list(per.values()))
        rows.append(dict(ligand=int(i),
                         maxlogD=float(vals.max()),
                         meanlogD=float(vals.mean()),
                         selectivity=float(vals.max() - vals.min()),  # log separation factor
                         n_metals=int(len(per))))
    strength = np.array([r["maxlogD"] for r in rows])
    selec = np.array([r["selectivity"] for r in rows])
    n = len(rows)

    # 1) ranking by strength != ranking by selectivity
    rho, _ = spearmanr(strength, selec)

    # 2) top-10 overlap: how many of the top-10 by logD are also top-10 by selectivity
    top_strength = set(np.argsort(-strength)[:10].tolist())
    top_selec = set(np.argsort(-selec)[:10].tolist())
    overlap = len(top_strength & top_selec)

    # 3) Pareto front in (strength, selectivity): non-dominated ligands
    pareto = []
    for a in range(n):
        dominated = any((strength[b] >= strength[a] and selec[b] >= selec[a] and
                         (strength[b] > strength[a] or selec[b] > selec[a]))
                        for b in range(n))
        if not dominated:
            pareto.append(a)

    res = dict(n_ligands=n, min_metals=MIN_METALS,
               spearman_strength_vs_selectivity=float(rho),
               top10_overlap=int(overlap),
               n_pareto=len(pareto),
               selectivity_range=[float(selec.min()), float(selec.max())],
               points=[dict(maxlogD=float(s), selectivity=float(v),
                            pareto=(k in set(pareto)))
                       for k, (s, v) in enumerate(zip(strength, selec))])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"ligands with >={MIN_METALS} metals: {n}")
    print(f"Spearman(max logD, selectivity) = {rho:.3f}  (low => strength != selectivity)")
    print(f"top-10 overlap (strength vs selectivity) = {overlap}/10")
    print(f"Pareto-optimal ligands = {len(pareto)}")
    print(f"selectivity (log separation factor) range = [{selec.min():.2f},{selec.max():.2f}]")
    print("E6_DONE")


if __name__ == "__main__":
    main()
