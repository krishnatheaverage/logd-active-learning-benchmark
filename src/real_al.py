import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from real_data import load_pool
from al_core import run_al

def main(strategies=("random", "greedy", "uncertainty", "ucb", "ei", "hybrid"),
         seeds=range(6), budget=420, batch=15, n_seed=25,
         calib_map=None, out="results/real_al_results.json"):
    X, y, cols = load_pool()
    print(f"pool {X.shape}, log_D range [{y.min():.2f},{y.max():.2f}]", flush=True)
    rng = np.random.default_rng(0)
    test_idx = rng.choice(len(y), int(0.2*len(y)), replace=False)
    pool_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    Xp, yp, Xt, yt = X[pool_idx], y[pool_idx], X[test_idx], y[test_idx]
    res = {}
    for strat in strategies:
        calib = (calib_map or {}).get(strat, 0.0)
        runs = []
        for s in seeds:
            r = run_al(Xp, yp, strategy=strat, n_seed=n_seed, batch=batch,
                       budget=budget, calib=calib, seed=s, test_X=Xt, test_y=yt)
            runs.append(r)
        nq = runs[0]["n_queried"]
        r2 = np.array([r["test_r2"] for r in runs])
        rec = np.array([r["top10_recall"] for r in runs])
        # labels needed to reach R2>=0.85 (their DNN level)
        reach = []
        for r in runs:
            arr = np.array(r["test_r2"]); idx = np.where(arr >= 0.85)[0]
            reach.append(nq[idx[0]] if len(idx) else np.nan)
        res[strat] = dict(n_queried=nq,
                          r2_mean=r2.mean(0).tolist(), r2_std=r2.std(0).tolist(),
                          recall_mean=rec.mean(0).tolist(), recall_std=rec.std(0).tolist(),
                          labels_to_R2_0p85=float(np.nanmean(reach)))
        print(f"  {strat:12s} final R2 {r2.mean(0)[-1]:.3f}  top10-recall {rec.mean(0)[-1]:.2f}  "
              f"labels->R2=0.85: {np.nanmean(reach):.0f}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(pool=int(len(pool_idx)), n_test=int(len(test_idx)), results=res),
              open(out, "w"), indent=2)
    print("REAL_AL_DONE", flush=True)

if __name__ == "__main__":
    main(calib_map={"ucb": 0.25, "ei": 0.25, "hybrid": 0.25})
