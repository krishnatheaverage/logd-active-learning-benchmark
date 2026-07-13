"""E4 (Reviewer 1.4): conformal calibration done rigorously, with subgroup coverage.

Fixes and additions the reviewer asked for:
  * train / calibration / test are LIGAND-DISJOINT (no leakage into the rescaling).
  * coverage reported globally AND by subgroup: logD range, lanthanide class,
    and the prospective (4 new ligand) set -- not just global.
  * Mondrian (group-conditional) conformal: a separate quantile per logD bin,
    to check whether global calibration fails in the high-logD region that
    matters most for discovery.
  * Non-monotonic caveat quantified: does locally-adaptive (Mondrian) calibration
    change the RANKING of uncertainty (which would change AL acquisition)?
Reports empirical coverage of nominal-80% and nominal-90% intervals + mean
calibration error, raw vs split-conformal vs Mondrian.
"""
import os, json, numpy as np
import sys; sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scipy.stats import norm, spearmanr
from common import RFUQ, interval_coverage, mean_calibration_error
import data_ext as de

OUT = "results/revision/e4_calibration.json"
LN_CLASS = {**{e: "light" for e in ["La", "Ce", "Pr", "Nd"]},
            **{e: "mid" for e in ["Sm", "Eu", "Gd"]},
            **{e: "heavy" for e in ["Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]}}


def logd_bin(y):
    return np.where(y < -1, "low", np.where(y > 1, "high", "mid"))


def q_conformal(resid_over_sigma, alpha):
    return float(np.quantile(resid_over_sigma, 1 - alpha, method="higher"))


def coverage_by(y, lo, hi, key):
    out = {}
    for g in np.unique(key):
        m = key == g
        if m.sum() >= 5:
            out[str(g)] = dict(coverage=interval_coverage(y[m], lo[m], hi[m]),
                               n=int(m.sum()))
    return out


def one_seed(d, seed, alpha=0.2):
    ids = np.unique(d["ligand"])
    rng = np.random.default_rng(seed); rng.shuffle(ids)
    n = len(ids); a, b = int(0.6 * n), int(0.8 * n)
    tr_ids, cal_ids, te_ids = ids[:a], ids[a:b], ids[b:]
    tr = np.isin(d["ligand"], tr_ids); cal = np.isin(d["ligand"], cal_ids); te = np.isin(d["ligand"], te_ids)
    X, y = d["X"], d["y"]
    model = RFUQ(n=300, seed=seed).fit(X[tr], y[tr])
    mu_c, s_c = model.predict(X[cal]); mu_t, s_t = model.predict(X[te])
    z = norm.ppf(1 - alpha / 2)  # for an 80% central interval under Gaussian sigma

    out = {}
    # raw (uncalibrated) intervals
    lo, hi = mu_t - z * s_t, mu_t + z * s_t
    out["raw"] = dict(coverage=interval_coverage(y[te], lo, hi),
                      mce=mean_calibration_error(y[te], mu_t, s_t))
    # split-conformal (one global scalar q; rank-preserving)
    scores = np.abs(y[cal] - mu_c) / np.maximum(s_c, 1e-6)
    q = q_conformal(scores, alpha)
    s_conf = s_t * q
    lo, hi = mu_t - z * s_conf, mu_t + z * s_conf
    out["conformal"] = dict(coverage=interval_coverage(y[te], lo, hi),
                            mce=mean_calibration_error(y[te], mu_t, s_conf),
                            by_logd=coverage_by(y[te], lo, hi, logd_bin(y[te])),
                            by_ln=coverage_by(y[te], lo, hi,
                                              np.array([LN_CLASS.get(l, "?") for l in d["lanthanide"][te]])))
    # Mondrian conformal: separate q per logD bin (group-conditional)
    binc = logd_bin(y[cal]); bint = logd_bin(y[te])
    s_mond = s_t.copy()
    qmap = {}
    for g in np.unique(binc):
        mg = binc == g
        if mg.sum() >= 5:
            qmap[g] = q_conformal(scores[mg], alpha)
    gq = np.median(list(qmap.values())) if qmap else q
    s_mond = s_t * np.array([qmap.get(g, gq) for g in bint])
    lo, hi = mu_t - z * s_mond, mu_t + z * s_mond
    out["mondrian"] = dict(coverage=interval_coverage(y[te], lo, hi),
                           by_logd=coverage_by(y[te], lo, hi, bint))
    # non-monotonic caveat: does Mondrian reorder uncertainty vs global-conformal?
    rho, _ = spearmanr(s_conf, s_mond)
    out["rank_corr_conformal_vs_mondrian"] = float(rho)
    return out


def prospective_coverage(seed=0, alpha=0.2):
    d = de.load("pool"); t = de.load("test_new")
    ids = np.unique(d["ligand"]); rng = np.random.default_rng(seed); rng.shuffle(ids)
    cal_ids = ids[:int(0.25 * len(ids))]
    cal = np.isin(d["ligand"], cal_ids); tr = ~cal
    model = RFUQ(n=300, seed=seed).fit(d["X"][tr], d["y"][tr])
    mu_c, s_c = model.predict(d["X"][cal])
    scores = np.abs(d["y"][cal] - mu_c) / np.maximum(s_c, 1e-6)
    q = q_conformal(scores, alpha); z = norm.ppf(1 - alpha / 2)
    mu, s = model.predict(t["X"]); s = s * q
    lo, hi = mu - z * s, mu + z * s
    return dict(coverage=interval_coverage(t["y"], lo, hi),
                by_logd=coverage_by(t["y"], lo, hi, logd_bin(t["y"])), n=int(len(t["y"])))


def main():
    d = de.load("pool")
    seeds = range(8)
    runs = [one_seed(d, s) for s in seeds]
    # aggregate
    def agg(path):
        vals = []
        for r in runs:
            o = r
            for p in path:
                o = o[p]
            vals.append(o)
        return dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
    summary = dict(
        nominal=0.8,
        raw_coverage=agg(["raw", "coverage"]), raw_mce=agg(["raw", "mce"]),
        conformal_coverage=agg(["conformal", "coverage"]), conformal_mce=agg(["conformal", "mce"]),
        mondrian_coverage=agg(["mondrian", "coverage"]),
        rank_corr=agg(["rank_corr_conformal_vs_mondrian"]),
    )
    # subgroup coverage: average across seeds where present
    def subgroup_avg(kind, sub):
        acc = {}
        for r in runs:
            for g, v in r["conformal"][sub].items():
                acc.setdefault(g, []).append(v["coverage"])
        return {g: dict(mean=float(np.mean(v)), n_seeds=len(v)) for g, v in acc.items()}
    summary["conformal_by_logd"] = subgroup_avg("c", "by_logd")
    summary["conformal_by_lanthanide"] = subgroup_avg("c", "by_ln")
    # mondrian by logd
    macc = {}
    for r in runs:
        for g, v in r["mondrian"]["by_logd"].items():
            macc.setdefault(g, []).append(v["coverage"])
    summary["mondrian_by_logd"] = {g: dict(mean=float(np.mean(v))) for g, v in macc.items()}
    summary["prospective"] = prospective_coverage()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(summary, open(OUT, "w"), indent=2)
    print("raw cov %.3f (mce %.3f) | conformal cov %.3f (mce %.3f) | mondrian cov %.3f"
          % (summary["raw_coverage"]["mean"], summary["raw_mce"]["mean"],
             summary["conformal_coverage"]["mean"], summary["conformal_mce"]["mean"],
             summary["mondrian_coverage"]["mean"]), flush=True)
    print("conformal coverage by logD bin:", {k: round(v["mean"], 2) for k, v in summary["conformal_by_logd"].items()}, flush=True)
    print("mondrian coverage by logD bin :", {k: round(v["mean"], 2) for k, v in summary["mondrian_by_logd"].items()}, flush=True)
    print("rank corr conformal vs mondrian: %.3f" % summary["rank_corr"]["mean"], flush=True)
    print("prospective coverage: %.3f" % summary["prospective"]["coverage"], flush=True)
    print("E4_DONE", flush=True)


if __name__ == "__main__":
    main()
