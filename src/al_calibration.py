import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy.stats import norm
from real_data import load_pool
from al_core import EnsembleSurrogate, run_al

X, y, cols = load_pool()

def coverage_experiment(seeds=range(10), nominals=(0.5, 0.8, 0.9)):
    rows = {n: {"uncalibrated": [], "conformal": []} for n in nominals}
    for s in seeds:
        rng = np.random.default_rng(s); idx = rng.permutation(len(y))
        ntr, ncal = int(0.6*len(y)), int(0.2*len(y))
        tr, cal, te = idx[:ntr], idx[ntr:ntr+ncal], idx[ntr+ncal:]
        sur = EnsembleSurrogate(seed=s).fit(X[tr], y[tr])
        m_te, s_te = sur.predict(X[te])
        m_cal, s_cal = sur.predict(X[cal])
        cal_scores = np.abs(y[cal] - m_cal) / np.maximum(s_cal, 1e-6)
        for nom in nominals:
            half = norm.ppf(0.5 + nom/2)                      # uncalibrated Gaussian interval
            cov_u = np.mean(np.abs(y[te]-m_te) <= half*s_te)
            q = np.quantile(cal_scores, nom)                  # conformal interval
            cov_c = np.mean(np.abs(y[te]-m_te) <= q*s_te)
            rows[nom]["uncalibrated"].append(cov_u); rows[nom]["conformal"].append(cov_c)
    out = {}
    print("Interval coverage (target = nominal):")
    print(f"  {'nominal':>8s} {'uncalibrated':>14s} {'conformal':>12s}")
    for nom in nominals:
        u = np.mean(rows[nom]["uncalibrated"]); c = np.mean(rows[nom]["conformal"])
        out[str(nom)] = dict(uncalibrated=float(u), conformal=float(c))
        print(f"  {nom:8.2f} {u:14.3f} {c:12.3f}")
    # calibration error (mean |coverage - nominal|)
    ece_u = np.mean([abs(out[str(n)]["uncalibrated"]-n) for n in nominals])
    ece_c = np.mean([abs(out[str(n)]["conformal"]-n) for n in nominals])
    out["calib_error"] = dict(uncalibrated=float(ece_u), conformal=float(ece_c))
    print(f"  mean calibration error: uncalibrated {ece_u:.3f}  vs  conformal {ece_c:.3f}")
    return out

def calib_ablation_al(strategies=("ucb", "ei", "hybrid"), seeds=range(5),
                      budget=300, batch=15, n_seed=25):
    rng = np.random.default_rng(0)
    test_idx = rng.choice(len(y), int(0.2*len(y)), replace=False)
    pool_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    Xp, yp, Xt, yt = X[pool_idx], y[pool_idx], X[test_idx], y[test_idx]
    res = {}
    print("\nAL with vs without conformal calibration (final R2 / recall):")
    for strat in strategies:
        for tag, cal in (("uncalib", 0.0), ("conformal", 0.25)):
            runs = [run_al(Xp, yp, strategy=strat, n_seed=n_seed, batch=batch,
                           budget=budget, calib=cal, seed=s, test_X=Xt, test_y=yt) for s in seeds]
            r2 = np.mean([r["test_r2"][-1] for r in runs])
            rec = np.mean([r["top10_recall"][-1] for r in runs])
            res[f"{strat}_{tag}"] = dict(final_r2=float(r2), recall=float(rec))
            print(f"  {strat:8s} {tag:10s} R2={r2:.3f} recall={rec:.2f}")
    return res

if __name__ == "__main__":
    out = dict(coverage=coverage_experiment(), al_ablation=calib_ablation_al())
    json.dump(out, open("results/calibration.json", "w"), indent=2)
    print("CALIBRATION_DONE")
