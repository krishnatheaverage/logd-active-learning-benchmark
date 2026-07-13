"""Extended data loader for the ACS Omega revision.

Recovers, from the released Liu et al. feature matrix (no SMILES are distributed),
the grouping structure the reviewer asked for:
  * ligand identity   -> unique ECFP(Morgan) bit-vector  (93 ligands)
  * lanthanide identity-> metal-descriptor block / atomic number (14 metals)
  * publication source -> the integer `reference` column (45 sources)
  * experimental conditions -> the 21 condition columns (concn, diluent, T, ...)
and an ECFP-Tanimoto ligand-family clustering used as a scaffold-split proxy
(exact Bemis-Murcko scaffolds are NOT recoverable without SMILES; stated in-paper).

All feature matrices are identical to the original benchmark (drop log_D + reference,
coerce numeric, fill 0) so numbers stay comparable to the R2=0.94 baseline.
"""
from __future__ import annotations
import numpy as np, pandas as pd

XLSX = "data/au2c00122_si_002.xlsx"
DROP = ["log_D", "reference"]

CONDITION_COLS = [
    'ligand c.c./mM', 'volume ratio of solvent a', 'volume ratio of solvent b',
    'Molar mass(a)', 'density(a)', 'boiling point(a)', 'melting point(a)',
    'Dipole moment(a)', 'Solubility in water(a)', 'log P(a)',
    'Molar mass(b)', 'density(b)', 'boiling point(b)', 'melting point(b)',
    'Dipole moment(b)', 'Solubility in water(b)', 'log P(b)',
    'aicd dipole', 'acid c.c./M', 'temperature', 'Ln c.c./mM',
]

# atomic number -> lanthanide symbol (from the 'lanthanides descriptors' sheet)
Z_TO_LN = {57:'La',58:'Ce',59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',
           64:'Gd',65:'Tb',66:'Dy',67:'Ho',68:'Er',69:'Tm',70:'Yb',71:'Lu'}


def _blocks(cols):
    ecfp  = [c for c in cols if str(c).startswith("ECFP")]
    metal = [c for c in cols if "metal" in str(c).lower()]
    cond  = [c for c in cols if str(c).strip() in {s.strip() for s in CONDITION_COLS}]
    return ecfp, metal, cond


def _features(df):
    """Model feature matrix: same recipe as the original benchmark."""
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.values.astype(float), list(X.columns)


def _atomic_number_col(metal_cols):
    for c in metal_cols:
        if "atomic number" in str(c).lower():
            return c
    raise KeyError("atomic-number column not found in metal block")


def load_frame(split="pool"):
    """Return the raw dataframe for a split. split in {train,val,pool,test_new}."""
    sheets = {"train": "training set", "val": "validation set",
              "test_new": "new ligands 1-4 as the test set"}
    if split == "pool":
        return pd.concat([pd.read_excel(XLSX, sheet_name=sheets["train"]),
                          pd.read_excel(XLSX, sheet_name=sheets["val"])],
                         ignore_index=True)
    return pd.read_excel(XLSX, sheet_name=sheets[split])


def load(split="pool"):
    """Feature matrix + target + group structure for a split.

    Returns dict with:
      X, y, cols, ligand (int id), lanthanide (symbol), source (int), cond (DataFrame)
    """
    df = load_frame(split)
    cols = list(df.columns)
    ecfp, metal, cond = _blocks(cols)
    X, feat_cols = _features(df)
    y = df["log_D"].astype(float).values

    # ligand identity = unique ECFP bit-vector (binarised for a stable key)
    ecfp_bits = (df[ecfp].values > 0).astype(np.int8)
    lig_str = pd.Series([b.tobytes() for b in ecfp_bits])
    ligand = lig_str.astype("category").cat.codes.values

    # lanthanide identity via atomic number
    zcol = _atomic_number_col(metal)
    lanthanide = np.array([Z_TO_LN.get(int(round(z)), f"Z{int(round(z))}")
                           for z in df[zcol].values])

    source = (df["reference"].astype(int).values if "reference" in df.columns
              else np.full(len(df), -1))

    return dict(X=X, y=y, cols=feat_cols, ligand=ligand, lanthanide=lanthanide,
                source=source, cond=df[cond].reset_index(drop=True),
                ecfp_bits=ecfp_bits, df=df)


def unique_ligand_fingerprints(d):
    """One representative ECFP bit-vector per unique ligand id (sorted by id)."""
    ids = np.unique(d["ligand"])
    reps = np.stack([d["ecfp_bits"][d["ligand"] == i][0] for i in ids])
    return ids, reps


def tanimoto_distance_matrix(bits):
    """Pairwise Tanimoto (Jaccard) distance for binary fingerprint rows."""
    b = bits.astype(bool)
    inter = b.astype(int) @ b.astype(int).T
    popc = b.sum(1)
    union = popc[:, None] + popc[None, :] - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return 1.0 - sim


def ligand_family_clusters(d, sim_threshold=0.35):
    """Cluster ligands into families by ECFP-Tanimoto (agglomerative, average
    linkage). Returns a per-row cluster label. sim_threshold is Tanimoto
    similarity; families merge when average similarity >= threshold.
    Proxy for a scaffold split (no SMILES -> no exact Murcko scaffold)."""
    from sklearn.cluster import AgglomerativeClustering
    ids, reps = unique_ligand_fingerprints(d)
    D = tanimoto_distance_matrix(reps)
    n = len(ids)
    if n < 2:
        lab_by_id = {ids[0]: 0}
    else:
        cl = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                     linkage="average",
                                     distance_threshold=1.0 - sim_threshold)
        labs = cl.fit_predict(D)
        lab_by_id = {i: int(l) for i, l in zip(ids, labs)}
    return np.array([lab_by_id[i] for i in d["ligand"]]), len(set(lab_by_id.values()))


if __name__ == "__main__":
    d = load("pool")
    print("pool:", d["X"].shape, "features")
    print("unique ligands:", len(np.unique(d["ligand"])))
    print("unique lanthanides:", sorted(set(d["lanthanide"])))
    print("unique sources:", len(np.unique(d["source"])))
    print("condition cols:", list(d["cond"].columns))
    fam, nfam = ligand_family_clusters(d)
    print(f"ligand families (Tanimoto>=0.35): {nfam} clusters over "
          f"{len(np.unique(d['ligand']))} ligands")
    import collections
    print("family sizes (by ligand):",
          sorted(collections.Counter(
              [fam[d['ligand']==i][0] for i in np.unique(d['ligand'])]).values(),
              reverse=True)[:15])
    # prospective
    t = load("test_new")
    print("prospective:", t["X"].shape, "ligands", len(np.unique(t["ligand"])),
          "lanthanides", len(set(t["lanthanide"])))
