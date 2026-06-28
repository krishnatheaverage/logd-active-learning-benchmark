import numpy as np, pandas as pd

XLSX = "data/au2c00122_si_002.xlsx"
DROP = ["log_D", "reference"]

def _clean(df):
    y = df["log_D"].astype(float).values
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.values.astype(float), y, list(X.columns)

def load(split="train"):
    sheet = {"train": "training set", "val": "validation set",
             "test_new": "new ligands 1-4 as the test set"}[split]
    return _clean(pd.read_excel(XLSX, sheet_name=sheet))

def load_pool():
    Xtr, ytr, cols = load("train")
    Xva, yva, _ = load("val")
    X = np.vstack([Xtr, Xva]); y = np.concatenate([ytr, yva])
    return X, y, cols

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    Xtr, ytr, cols = load("train")
    Xva, yva, _ = load("val")
    print(f"train {Xtr.shape}  val {Xva.shape}  features={len(cols)}")
    print(f"y(train) range [{ytr.min():.2f},{ytr.max():.2f}] mean {ytr.mean():.2f}")
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    p = rf.predict(Xva)
    print(f"RF baseline on val: R2={r2_score(yva,p):.3f}  RMSE={mean_squared_error(yva,p)**0.5:.3f}")
    print("(Liu et al. DNN reported R2=0.85, RMSE=0.53)")
