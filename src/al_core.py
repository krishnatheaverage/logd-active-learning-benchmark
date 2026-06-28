import os, sys, json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_generative"))
from descriptors import featurize_many

def load_pool(path="results/generative/vmin_labels.json", target="donor"):
    raw = json.load(open(path))
    smi = [s for s, v in raw.items() if isinstance(v, dict) and "vmin" in v]
    y = np.array([-raw[s]["vmin"] for s in smi])
    X, keep = featurize_many(smi)
    y = np.array([y[smi.index(s)] for s in keep])
    return np.asarray(X, float), y, keep

class EnsembleSurrogate:
    def __init__(self, n_trees=300, seed=0):
        self.rf = RandomForestRegressor(n_estimators=n_trees, max_depth=None,
                                        min_samples_leaf=2, n_jobs=-1, random_state=seed)
        self.mu = self.sd = None; self.q = 1.0
    def fit(self, X, y, calib_frac=0.0, rng=None):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        if calib_frac > 0 and len(y) > 20:
            rng = rng or np.random.default_rng(0)
            idx = rng.permutation(len(y)); ncal = max(5, int(calib_frac*len(y)))
            cal, tr = idx[:ncal], idx[ncal:]
            self.rf.fit(Xs[tr], y[tr])
            m, s = self._raw(Xs[cal])
            scores = np.abs(y[cal] - m) / np.maximum(s, 1e-6)
            self.q = float(np.quantile(scores, 0.8))
            self.rf.fit(Xs, y)
        else:
            self.rf.fit(Xs, y); self.q = 1.0
        return self
    def _raw(self, Xs):
        preds = np.stack([t.predict(Xs) for t in self.rf.estimators_])
        return preds.mean(0), preds.std(0)
    def predict(self, X):
        m, s = self._raw((X - self.mu) / self.sd)
        return m, s * self.q

ACQ = {
    "random":   lambda m, s, best, rng: rng.random(len(m)),
    "greedy":   lambda m, s, best, rng: m,
    "uncertainty": lambda m, s, best, rng: s,
    "ucb":      lambda m, s, best, rng: m + 1.5 * s,
    "ei":       lambda m, s, best, rng: _ei(m, s, best),
}

def _ei(m, s, best):
    from scipy.stats import norm
    s = np.maximum(s, 1e-9); z = (m - best) / s
    return (m - best) * norm.cdf(z) + s * norm.pdf(z)

def run_al(X, y, strategy="ucb", n_seed=8, batch=4, budget=None, calib=0.0, seed=0,
           test_X=None, test_y=None):
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    N = len(y); budget = budget or min(N, int(0.8*N))
    true_top = set(np.argsort(-y)[:10])
    revealed = list(rng.choice(N, n_seed, replace=False))
    best_curve, recall_curve, r2_curve, nq = [], [], [], []
    while len(revealed) < budget:
        rmask = np.ones(N, bool); rmask[revealed] = False
        unrev = np.where(rmask)[0]
        surr = None
        if strategy == "random":
            pick = unrev[np.argsort(-rng.random(len(unrev)))[:batch]]
        elif strategy == "hybrid":
            surr = EnsembleSurrogate(seed=seed).fit(X[revealed], y[revealed],
                                                    calib_frac=calib, rng=rng)
            m, s = surr.predict(X[unrev])
            n_exploit = batch // 2
            ex = list(unrev[np.argsort(-m)[:n_exploit]])          # exploit: design
            rest = [i for i in unrev if i not in set(ex)]
            sr = s[np.isin(unrev, rest)]
            ep = list(np.array(rest)[np.argsort(-sr)[:batch - n_exploit]])  # explore: prediction
            pick = np.array(ex + ep)
        else:
            surr = EnsembleSurrogate(seed=seed).fit(X[revealed], y[revealed],
                                                    calib_frac=calib, rng=rng)
            m, s = surr.predict(X[unrev]); best = y[revealed].max()
            score = ACQ[strategy](m, s, best, rng)
            pick = unrev[np.argsort(-score)[:batch]]
        revealed += list(pick)
        best_curve.append(float(y[revealed].max()))
        recall_curve.append(len(set(revealed) & true_top) / len(true_top))
        nq.append(len(revealed))
        if test_X is not None:
            sm = surr or EnsembleSurrogate(seed=seed).fit(X[revealed], y[revealed])
            r2_curve.append(float(r2_score(test_y, sm.predict(test_X)[0])))
    return dict(strategy=strategy, n_queried=nq, best=best_curve,
                top10_recall=recall_curve, test_r2=r2_curve)

def experiment(strategies=("random","greedy","uncertainty","ucb","ei"),
               seeds=range(10), calib_map=None, out="results/al_results.json"):
    X, y, smi = load_pool()
    print(f"pool: {len(y)} ligands, target range [{y.min():.3f},{y.max():.3f}]", flush=True)
    res = {}
    for strat in strategies:
        calib = (calib_map or {}).get(strat, 0.0)
        runs = [run_al(X, y, strategy=strat, calib=calib, seed=s) for s in seeds]
        nq = runs[0]["n_queried"]
        best = np.array([r["best"] for r in runs]); rec = np.array([r["top10_recall"] for r in runs])
        res[strat] = dict(n_queried=nq,
                          best_mean=best.mean(0).tolist(), best_std=best.std(0).tolist(),
                          recall_mean=rec.mean(0).tolist(), recall_std=rec.std(0).tolist())
        print(f"  {strat:12s} final top10-recall {rec.mean(0)[-1]:.2f}+/-{rec.std(0)[-1]:.2f}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(n_pool=len(y), results=res), open(out, "w"), indent=2)
    print("AL_EXPERIMENT_DONE", flush=True)
    return res

if __name__ == "__main__":
    experiment(calib_map={"ucb": 0.25, "ei": 0.25})
