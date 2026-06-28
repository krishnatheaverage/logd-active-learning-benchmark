import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "04_generative"))
from rdkit import Chem
from rdkit.Chem import AllChem
from descriptors import featurize
from al_core import run_al

FEAT = "data/lipo_features.npz"

def featurize_strong(smi, nbits=1024):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits))
    d = featurize(smi)
    if d is None or not np.all(np.isfinite(d)): return None
    return np.concatenate([fp, d])

def load():
    if os.path.exists(FEAT):
        d = np.load(FEAT); return d["X"], d["y"]
    df = pd.read_csv("data/lipophilicity.csv")
    X, y = [], []
    for smi, exp in zip(df["smiles"], df["exp"].astype(float)):
        v = featurize_strong(smi)
        if v is not None:
            X.append(v); y.append(exp)
    X, y = np.array(X), np.array(y)
    np.savez(FEAT, X=X, y=y)
    return X, y

def main(strategies=("random", "greedy", "uncertainty", "ucb", "ei", "hybrid"),
         seeds=range(6), budget=700, batch=25, n_seed=40,
         calib_map={"ucb": 0.25, "ei": 0.25, "hybrid": 0.25},
         out="results/lipo_al_results.json"):
    X, y = load()
    print(f"Lipophilicity: {X.shape} compounds, logD range [{y.min():.2f},{y.max():.2f}]", flush=True)
    rng = np.random.default_rng(0)
    test_idx = rng.choice(len(y), int(0.2*len(y)), replace=False)
    pool_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    Xp, yp, Xt, yt = X[pool_idx], y[pool_idx], X[test_idx], y[test_idx]
    res = {}
    for strat in strategies:
        calib = calib_map.get(strat, 0.0)
        runs = [run_al(Xp, yp, strategy=strat, n_seed=n_seed, batch=batch, budget=budget,
                       calib=calib, seed=s, test_X=Xt, test_y=yt) for s in seeds]
        nq = runs[0]["n_queried"]
        r2 = np.array([r["test_r2"] for r in runs]); rec = np.array([r["top10_recall"] for r in runs])
        res[strat] = dict(n_queried=nq, r2_mean=r2.mean(0).tolist(), r2_std=r2.std(0).tolist(),
                          recall_mean=rec.mean(0).tolist(), recall_std=rec.std(0).tolist())
        print(f"  {strat:12s} final R2 {r2.mean(0)[-1]:.3f}  top10-recall {rec.mean(0)[-1]:.2f}", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(pool=int(len(pool_idx)), n_test=int(len(test_idx)), results=res),
              open(out, "w"), indent=2)
    print("LIPO_AL_DONE", flush=True)

if __name__ == "__main__":
    main()
