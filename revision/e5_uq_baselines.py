"""E5 (Reviewer 1.5): broader model / UQ-estimator baselines.

The reviewer asked for more than a random forest: Gaussian process, quantile
regression forest, gradient boosting, a Bayesian NN, and a conformalized model.
We compare all of them on a LIGAND-DISJOINT split (train/calib/test) for:
  * predictive accuracy (R2, RMSE)
  * calibration (mean calibration error, raw and split-conformal 80% coverage)
  * uncertainty-RANKING quality: Spearman(|error|, sigma) and selective-prediction
    RMSE-AUC (does dropping high-sigma points cut error?)
  * discovery quality: enrichment factor of ranking the pool by predicted mean
This gives the "more insight into uncertainty ranking quality" the reviewer wanted.
"""
import os, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch; torch.set_num_threads(1)  # prevent torch/joblib deadlock
except Exception:
    pass
import sys; sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.stats import norm
from common import (RFUQ, QRF, GPUQ, GBMUQ, MCDropoutBNN, reg_metrics,
                    mean_calibration_error, spearman_error_sigma,
                    selective_rmse_curve, interval_coverage, enrichment_factor)
import data_ext as de

OUT = "results/revision/e5_uq_baselines.json"
FACTORY = {"RF": RFUQ, "QRF": QRF, "GP": GPUQ, "GBM": GBMUQ, "BNN": MCDropoutBNN}


def evaluate(model, Xtr, ytr, Xcal, ycal, Xte, yte, alpha=0.2):
    model.fit(Xtr, ytr)
    mu, sigma = model.predict(Xte)
    m = reg_metrics(yte, mu)
    # split-conformal on the calibration set
    mc, sc = model.predict(Xcal)
    scores = np.abs(ycal - mc) / np.maximum(sc, 1e-6)
    q = float(np.quantile(scores, 1 - alpha, method="higher"))
    z = norm.ppf(1 - alpha / 2)
    cov = interval_coverage(yte, mu - z * sigma * q, mu + z * sigma * q)
    _, auc = selective_rmse_curve(yte, mu, sigma)
    return dict(r2=m["r2"], rmse=m["rmse"],
                mce_raw=mean_calibration_error(yte, mu, sigma),
                conformal_cov80=float(cov),
                spearman_err_sigma=spearman_error_sigma(yte, mu, sigma),
                selective_rmse_auc=float(auc),
                enrichment=float(enrichment_factor(yte, np.argsort(-mu)[:10], k=10)))


def main():
    d = de.load("pool")
    X, y, lig = d["X"], d["y"], d["ligand"]
    ids = np.unique(lig)
    seeds = range(3)
    acc = {k: [] for k in FACTORY}
    for seed in seeds:
        rng = np.random.default_rng(seed); order = ids.copy(); rng.shuffle(order)
        a, b = int(0.55 * len(order)), int(0.75 * len(order))
        tr = np.isin(lig, order[:a]); cal = np.isin(lig, order[a:b]); te = np.isin(lig, order[b:])
        for name, F in FACTORY.items():
            try:
                r = evaluate(F(seed=seed), X[tr], y[tr], X[cal], y[cal], X[te], y[te])
                acc[name].append(r)
                print(f"seed {seed} {name:4s} R2 {r['r2']:.3f} RMSE {r['rmse']:.3f} "
                      f"cov80 {r['conformal_cov80']:.2f} rho(err,sig) {r['spearman_err_sigma']:.2f} "
                      f"EF {r['enrichment']:.1f}", flush=True)
            except Exception as e:
                print(f"seed {seed} {name} FAILED: {type(e).__name__}: {e}", flush=True)
    res = {}
    for name, rows in acc.items():
        if not rows:
            continue
        res[name] = {k: dict(mean=float(np.mean([r[k] for r in rows])),
                             sd=float(np.std([r[k] for r in rows])))
                     for k in rows[0]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(split="ligand-disjoint 55/20/25", seeds=len(list(seeds)), results=res),
              open(OUT, "w"), indent=2)
    print("E5_DONE", flush=True)


if __name__ == "__main__":
    main()
