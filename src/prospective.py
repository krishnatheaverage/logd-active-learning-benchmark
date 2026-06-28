import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

XLSX = "data/au2c00122_si_002.xlsx"

def sheet(name):
    return pd.read_excel(XLSX, sheet_name=name)

tr = pd.concat([sheet("training set"), sheet("validation set")], ignore_index=True)
te = sheet("new ligands 1-4 as the test set")
ycol = "log_D"
feat = [c for c in tr.columns if c not in (ycol, "reference")]
feat = [c for c in feat if c in te.columns]          # align to columns present in both
Xtr = tr[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
ytr = tr[ycol].astype(float).values
Xte = te[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
yte = te[ycol].astype(float).values
print(f"train {Xtr.shape}, prospective test {Xte.shape} ({len(yte)} measurements on the 4 new ligands)")

rf = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=0).fit(Xtr, ytr)
p = rf.predict(Xte)
out = dict(n_test=int(len(yte)),
           r2=float(r2_score(yte, p)),
           rmse=float(mean_squared_error(yte, p)**0.5),
           mae=float(mean_absolute_error(yte, p)),
           y_true=yte.tolist(), y_pred=p.tolist())
print(f"PROSPECTIVE (4 new, independently-synthesised ligands):")
print(f"  R2 = {out['r2']:.3f}   RMSE = {out['rmse']:.3f}   MAE = {out['mae']:.3f} log units")
print(f"  (Liu et al. reported these 4 as their external test; their DNN R2~0.85 in-domain)")
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/prospective.json", "w"), indent=2)
print("PROSPECTIVE_DONE")
