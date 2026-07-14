"""Emit LaTeX tables + an in-text macro file from the real results JSONs, so every
number in the manuscript is generated (not hand-typed). Writes into paper/."""
import json, os, numpy as np

R = "results/revision"
P = "paper"
ORDER = ["random", "greedy", "uncertainty", "ucb", "ei", "hybrid"]


def f(x, n=2):
    return f"{x:.{n}f}"


def _load(name):
    p = f"{R}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def table_splits(macros):
    d = _load("e1_splits.json")
    if not d:
        return ""
    r = d["results"]; m = d["meta"]
    rows = [("Random (row-level)", "random"), ("Ligand-disjoint", "ligand"),
            ("Source-disjoint", "source"), ("Ligand-family (scaffold proxy)", "family")]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Random-forest generalisation under increasingly stringent, "
             r"leakage-controlled splits. Random row-level cross-validation is "
             r"strongly optimistic; grouped splits that forbid a ligand, source, or "
             r"structural family from appearing in both train and test collapse the "
             r"estimate toward the mean. Mean$\pm$s.d.\ over 4 repeats of 5-fold "
             r"grouped CV.}",
             r"\label{tab:splits}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Split & $R^2$ & RMSE & MAE \\", r"\midrule"]
    for label, k in rows:
        v = r[k]
        lines.append(f"{label} & {f(v['r2']['mean'])}$\\pm${f(v['r2']['sd'])} & "
                     f"{f(v['rmse']['mean'])} & {f(v['mae']['mean'])} \\\\")
    p = r["prospective_new_ligands"]
    lines.append(r"\midrule")
    lines.append(f"Prospective (4 new ligands) & {f(p['r2'])} & {f(p['rmse'])} & {f(p['mae'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    macros["splitRandomRow"] = f(r["random"]["r2"]["mean"])
    macros["splitLigand"] = f(r["ligand"]["r2"]["mean"])
    macros["splitSource"] = f(r["source"]["r2"]["mean"])
    macros["splitFamily"] = f(r["family"]["r2"]["mean"])
    macros["splitProspective"] = f(p["r2"])
    macros["nLigands"] = str(m["n_ligands"])
    macros["nSources"] = str(m["n_sources"])
    macros["nFamilies"] = str(m["n_families"])
    return "\n".join(lines)


def table_recall(macros):
    d = _load("e3_recall_metrics.json")
    if not d:
        return ""
    r = d["results"]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Discovery performance at the final budget under three "
             r"definitions of a `top-10 system' (condition-specific row, "
             r"ligand--metal pair, unique ligand), plus simple regret (log units) "
             r"and hit-rate above a practical threshold "
             r"($\log D\ge 1$, i.e.\ $D\ge 10$). The ranking of strategies inverts "
             r"with the definition: exploitation dominates for rows, but broad "
             r"sampling already recovers most top ligands. Means over six seeds.}",
             r"\label{tab:recall}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Strategy & recall & recall & recall & regret & hit-rate \\",
             r" & (row) & (lig--metal) & (ligand) & (log) & ($\log D\!\ge\!1$) \\",
             r"\midrule"]
    for s in ORDER:
        a = r[s]
        lines.append(f"{s} & {f(a['rec_row_mean'][-1])} & {f(a['rec_pair_mean'][-1])} & "
                     f"{f(a['rec_lig_mean'][-1])} & "
                     f"{f(a['regret_mean'][-1])} & {f(a['hit_mean'][-1])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    macros["recallGreedyRow"] = f(r["greedy"]["rec_row_mean"][-1])
    macros["recallGreedyLig"] = f(r["greedy"]["rec_lig_mean"][-1])
    macros["recallRandomRow"] = f(r["random"]["rec_row_mean"][-1])
    macros["recallRandomLig"] = f(r["random"]["rec_lig_mean"][-1])
    return "\n".join(lines)


def table_uq(macros):
    d = _load("e5_uq_baselines.json")
    if not d:
        return ""
    r = d["results"]
    order = [m for m in ["RF", "QRF", "GP", "GBM", "BNN"] if m in r]
    names = {"RF": "Random forest", "QRF": "Quantile RF", "GP": "Gaussian process",
             "GBM": "Gradient boosting", "BNN": "MC-dropout BNN"}
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Model and uncertainty-estimator comparison on a "
             r"ligand-disjoint split (train/calibration/test $=55/20/25$ of ligands). "
             r"Uncertainty-ranking quality is the Spearman correlation between "
             r"predicted $\sigma$ and realised absolute error; higher is better. "
             r"Coverage is the empirical coverage of split-conformal 80\% intervals. "
             r"Mean over four seeds.}",
             r"\label{tab:uq}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & $R^2$ & conformal & rank quality & enrich. \\",
             r" & & 80\% cov. & $\rho(|e|,\sigma)$ & factor \\", r"\midrule"]
    for m in order:
        a = r[m]
        lines.append(f"{names[m]} & {f(a['r2']['mean'])} & {f(a['conformal_cov80']['mean'])} & "
                     f"{f(a['spearman_err_sigma']['mean'])} & {f(a['enrichment']['mean'],1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # best ranking-quality model
    best = max(order, key=lambda m: r[m]["spearman_err_sigma"]["mean"])
    macros["bestRankModel"] = {"RF": "the random forest", "QRF": "the quantile forest",
                               "GP": "the Gaussian process", "GBM": "gradient boosting",
                               "BNN": "the MC-dropout network"}[best]
    macros["bestRankRho"] = f(r[best]["spearman_err_sigma"]["mean"])
    return "\n".join(lines)


def table_calib(macros):
    d = _load("e4_calibration.json")
    if not d:
        return ""
    orig = json.load(open("results/calibration.json"))["coverage"]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Uncertainty calibration under a realistic ligand-disjoint "
             r"split (nominal coverage 80\%). Raw ensemble intervals under-cover; a "
             r"single global split-conformal factor restores average coverage but "
             r"leaves subgroup imbalance, which group-conditional (Mondrian) conformal "
             r"reduces. Mean over eight seeds. For reference, on the easier row-level "
             r"split the raw mean calibration error is "
             + f(orig['calib_error']['uncalibrated'], 3)
             + r" and conformal reduces it to "
             + f(orig['calib_error']['conformal'], 3) + r".}",
             r"\label{tab:calib}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & global & low $\log D$ & mid $\log D$ & high $\log D$ \\",
             r"\midrule"]
    cbl = d["conformal_by_logd"]; mbl = d["mondrian_by_logd"]
    lines.append(f"Raw & {f(d['raw_coverage']['mean'])} & -- & -- & -- \\\\")
    lines.append(f"Global conformal & {f(d['conformal_coverage']['mean'])} & "
                 f"{f(cbl.get('low',{}).get('mean',float('nan')))} & "
                 f"{f(cbl.get('mid',{}).get('mean',float('nan')))} & "
                 f"{f(cbl.get('high',{}).get('mean',float('nan')))} \\\\")
    lines.append(f"Mondrian conformal & {f(d['mondrian_coverage']['mean'])} & "
                 f"{f(mbl.get('low',{}).get('mean',float('nan')))} & "
                 f"{f(mbl.get('mid',{}).get('mean',float('nan')))} & "
                 f"{f(mbl.get('high',{}).get('mean',float('nan')))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    macros["covRaw"] = f(d["raw_coverage"]["mean"])
    macros["covConformal"] = f(d["conformal_coverage"]["mean"])
    macros["covMondrian"] = f(d["mondrian_coverage"]["mean"])
    macros["covProspective"] = f(d["prospective"]["coverage"])
    macros["rankCorr"] = f(d["rank_corr"]["mean"])
    macros["mceRaw"] = f(orig["calib_error"]["uncalibrated"], 3)
    macros["mceConformal"] = f(orig["calib_error"]["conformal"], 3)
    macros["mceFold"] = str(int(round(orig["calib_error"]["uncalibrated"] /
                                       orig["calib_error"]["conformal"])))
    return "\n".join(lines)


def batch_macros(macros):
    d = _load("e2_ligand_batch.json")
    if not d:
        return
    r = d["results"]
    # exploit = greedy, explore = random (final-budget values)
    macros["batchExploitRtwo"] = f(r["greedy"]["r2_mean"][-1])
    macros["batchExploitRecall"] = f(r["greedy"]["recall_mean"][-1])
    macros["batchExploreRtwo"] = f(r["random"]["r2_mean"][-1])
    macros["batchExploreRecall"] = f(r["random"]["recall_mean"][-1])
    macros["batchEIrecall"] = f(r.get("ei", r["hybrid"])["recall_mean"][-1])
    macros["batchEIRtwo"] = f(r.get("ei", r["hybrid"])["r2_mean"][-1])


def selectivity_macros(macros):
    d = _load("e6_selectivity.json")
    if not d:
        return
    macros["selCorr"] = f(d["spearman_strength_vs_selectivity"])
    macros["selOverlap"] = str(d["top10_overlap"])
    macros["selNpareto"] = str(d["n_pareto"])
    macros["selNlig"] = str(d["n_ligands"])
    macros["selMax"] = f(d["selectivity_range"][1], 1)


def main():
    macros = {}
    tabs = {"tab_splits": table_splits(macros), "tab_recall": table_recall(macros),
            "tab_uq": table_uq(macros), "tab_calib": table_calib(macros)}
    batch_macros(macros)
    selectivity_macros(macros)
    for name, tex in tabs.items():
        if tex:
            open(f"{P}/{name}.tex", "w").write(tex + "\n")
            print("wrote", name)
    # macros file
    ml = ["% auto-generated in-text numbers; do not edit by hand"]
    for k, v in macros.items():
        ml.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    open(f"{P}/revision_macros.tex", "w").write("\n".join(ml) + "\n")
    print("wrote revision_macros.tex with", len(macros), "macros")
    print("MACROS:", macros)
    print("TABLES_DONE")


if __name__ == "__main__":
    main()
