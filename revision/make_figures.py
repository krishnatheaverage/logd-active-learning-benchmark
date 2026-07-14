"""Generate all revision figures from the real results JSONs (plotnine / ggplot
look, greyscale, series distinguished by linetype/shape, matching house style)."""
import json, numpy as np, pandas as pd
from plotnine import (ggplot, aes, geom_line, geom_ribbon, geom_point, geom_col,
                      geom_hline, geom_errorbar, facet_wrap, theme_bw, theme, labs,
                      scale_linetype_manual, scale_shape_manual, scale_fill_grey,
                      element_text, element_rect, position_dodge)

R = "results/revision"
F = "figures"
ORDER = ["random", "greedy", "uncertainty", "ucb", "ei", "hybrid"]
LT = ["solid", "dashed", "dotted", "dashdot", (0, (6, 2)), (0, (3, 1, 1, 1, 1, 1))]
SH = ["o", "s", "^", "D", "v", "P"]


def fig_splits():
    d = json.load(open(f"{R}/e1_splits.json"))["results"]
    rows = []
    name = {"random": "random\n(row)", "ligand": "ligand", "source": "source",
            "family": "family\n(scaffold proxy)"}
    for k in ["random", "ligand", "source", "family"]:
        rows.append(dict(split=name[k], r2=d[k]["r2"]["mean"], sd=d[k]["r2"]["sd"]))
    p = d["prospective_new_ligands"]
    rows.append(dict(split="prospective\n(4 new)", r2=p["r2"], sd=0.0))
    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(df["split"], categories=[r["split"] for r in rows], ordered=True)
    g = (ggplot(df, aes("split", "r2"))
         + geom_col(fill="#B0B0B0", color="black", width=0.65)
         + geom_errorbar(aes(ymin="r2-sd", ymax="r2+sd"), width=0.2)
         + geom_hline(yintercept=0, size=0.4)
         + labs(x="train/test split", y="RF test R²",
                title="Grouped splits collapse the optimistic row-level R²")
         + theme_bw(base_size=11)
         + theme(figure_size=(6.2, 3.4), plot_title=element_text(weight="bold", size=10)))
    g.save(f"{F}/splits.pdf", dpi=300, verbose=False)
    g.save(f"{F}/splits.png", dpi=150, verbose=False)
    print("wrote splits")


def _curve_df(d, xkey, keys):
    rows = []
    lab = {"r2": "Prediction: test R²", "recall": "Design: top-k recall"}
    for s in ORDER:
        if s not in d:
            continue
        r = d[s]
        for i, x in enumerate(r[xkey]):
            for key in keys:
                m = r[f"{key}_mean"][i]; sd = r[f"{key}_std"][i]
                rows.append(dict(strategy=s, x=x, metric=lab[key], value=m, lo=m - sd, hi=m + sd))
    df = pd.DataFrame(rows)
    df["strategy"] = pd.Categorical(df["strategy"], categories=ORDER, ordered=True)
    return df


def fig_ligand_batch():
    d = json.load(open(f"{R}/e2_ligand_batch.json"))["results"]
    df = _curve_df(d, "n_lig", ["r2", "recall"])
    g = (ggplot(df, aes("x", "value"))
         + geom_ribbon(aes(ymin="lo", ymax="hi", group="strategy"), fill="#B3B3B3", alpha=0.25)
         + geom_line(aes(linetype="strategy"), size=0.7)
         + geom_point(aes(shape="strategy"), size=1.3)
         + facet_wrap("metric", scales="free_y")
         + scale_linetype_manual(values=LT) + scale_shape_manual(values=SH)
         + labs(x="number of ligands acquired (each reveals ~14 measurements)",
                y="", linetype="", shape="",
                title="Ligand-batch acquisition: the dichotomy persists")
         + theme_bw(base_size=11)
         + theme(figure_size=(8.2, 3.4), legend_position="right",
                 plot_title=element_text(weight="bold", size=10),
                 strip_background=element_rect(fill="#E6E6E6"),
                 strip_text=element_text(weight="bold")))
    g.save(f"{F}/ligand_batch.pdf", dpi=300, verbose=False)
    g.save(f"{F}/ligand_batch.png", dpi=150, verbose=False)
    print("wrote ligand_batch")


def fig_recall_defs():
    d = json.load(open(f"{R}/e3_recall_metrics.json"))["results"]
    lab = {"rec_row": "row", "rec_pair": "ligand-metal", "rec_lig": "ligand"}
    rows = []
    for s in ORDER:
        for k, nm in lab.items():
            rows.append(dict(strategy=s, definition=nm, recall=d[s][f"{k}_mean"][-1]))
    df = pd.DataFrame(rows)
    df["strategy"] = pd.Categorical(df["strategy"], categories=ORDER, ordered=True)
    df["definition"] = pd.Categorical(df["definition"], categories=["row", "ligand-metal", "ligand"], ordered=True)
    g = (ggplot(df, aes("strategy", "recall", fill="definition"))
         + geom_col(position=position_dodge(width=0.8), width=0.75, color="black")
         + scale_fill_grey(start=0.35, end=0.9)
         + labs(x="acquisition strategy", y="final top-10 recall", fill="'top' defined by",
                title="Top-k recall depends on how a 'top system' is defined")
         + theme_bw(base_size=11)
         + theme(figure_size=(7.2, 3.4), legend_position="right",
                 plot_title=element_text(weight="bold", size=10),
                 axis_text_x=element_text(rotation=20, ha="right")))
    g.save(f"{F}/recall_defs.pdf", dpi=300, verbose=False)
    g.save(f"{F}/recall_defs.png", dpi=150, verbose=False)
    print("wrote recall_defs")


def fig_calibration_subgroups():
    d = json.load(open(f"{R}/e4_calibration.json"))
    rows = []
    for meth, key in [("raw", None), ("conformal", "conformal_by_logd"), ("mondrian", "mondrian_by_logd")]:
        if meth == "raw":
            rows.append(dict(method="raw", bin="global", coverage=d["raw_coverage"]["mean"]))
        for b, v in d.get(key, {}).items():
            rows.append(dict(method=meth, bin=b, coverage=v["mean"]))
    rows.append(dict(method="conformal", bin="global", coverage=d["conformal_coverage"]["mean"]))
    rows.append(dict(method="mondrian", bin="global", coverage=d["mondrian_coverage"]["mean"]))
    df = pd.DataFrame(rows)
    order_bin = ["global", "low", "mid", "high"]
    df = df[df["bin"].isin(order_bin)]
    df["bin"] = pd.Categorical(df["bin"], categories=order_bin, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=["raw", "conformal", "mondrian"], ordered=True)
    g = (ggplot(df, aes("bin", "coverage", fill="method"))
         + geom_col(position=position_dodge(width=0.8), width=0.75, color="black")
         + geom_hline(yintercept=0.8, linetype="dashed", size=0.5)
         + scale_fill_grey(start=0.3, end=0.9)
         + labs(x="logD region", y="empirical coverage of 80% interval", fill="",
                title="Subgroup coverage: global conformal can miss the high-logD region")
         + theme_bw(base_size=11)
         + theme(figure_size=(6.6, 3.4), legend_position="right",
                 plot_title=element_text(weight="bold", size=9)))
    g.save(f"{F}/calibration_subgroups.pdf", dpi=300, verbose=False)
    g.save(f"{F}/calibration_subgroups.png", dpi=150, verbose=False)
    print("wrote calibration_subgroups")


def fig_selectivity():
    d = json.load(open(f"{R}/e6_selectivity.json"))
    df = pd.DataFrame(d["points"])
    df["frontier"] = np.where(df["pareto"], "Pareto-optimal", "dominated")
    g = (ggplot(df, aes("maxlogD", "selectivity"))
         + geom_point(aes(shape="frontier", size="frontier"), fill="white", stroke=0.6)
         + scale_shape_manual(values=["o", "D"])
         + labs(x="extraction strength (max $\\log D$ over lanthanides)",
                y="selectivity (log separation factor,\nmax$-$min $\\log D$)",
                shape="", size="",
                title="High $\\log D$ $\\neq$ high selectivity (Spearman "
                      + f"{d['spearman_strength_vs_selectivity']:.2f}, "
                      + f"top-10 overlap {d['top10_overlap']}/10)")
         + theme_bw(base_size=11)
         + theme(figure_size=(6.4, 3.6), legend_position="right",
                 plot_title=element_text(weight="bold", size=9)))
    try:
        from plotnine import scale_size_manual
        g = g + scale_size_manual(values=[1.6, 3.2])
    except Exception:
        pass
    g.save(f"{F}/selectivity.pdf", dpi=300, verbose=False)
    g.save(f"{F}/selectivity.png", dpi=150, verbose=False)
    print("wrote selectivity")


def fig_uq_models():
    d = json.load(open(f"{R}/e5_uq_baselines.json"))["results"]
    rows = []
    metr = {"conformal_cov80": "conformal 80% coverage",
            "spearman_err_sigma": "uncertainty ranking\n(Spearman err vs σ)"}
    for m in d:
        for k, nm in metr.items():
            rows.append(dict(model=m, metric=nm, value=d[m][k]["mean"], sd=d[m][k]["sd"]))
    df = pd.DataFrame(rows)
    g = (ggplot(df, aes("model", "value"))
         + geom_col(fill="#B0B0B0", color="black", width=0.65)
         + geom_errorbar(aes(ymin="value-sd", ymax="value+sd"), width=0.2)
         + facet_wrap("metric", scales="free_y")
         + labs(x="model", y="",
                title="Uncertainty quality across estimators")
         + theme_bw(base_size=11)
         + theme(figure_size=(8.0, 3.4), plot_title=element_text(weight="bold", size=10),
                 strip_background=element_rect(fill="#E6E6E6"),
                 strip_text=element_text(weight="bold")))
    g.save(f"{F}/uq_models.pdf", dpi=300, verbose=False)
    g.save(f"{F}/uq_models.png", dpi=150, verbose=False)
    print("wrote uq_models")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["splits", "ligand_batch", "recall_defs", "calibration_subgroups", "uq_models", "selectivity"]
    fns = dict(splits=fig_splits, ligand_batch=fig_ligand_batch, recall_defs=fig_recall_defs,
               calibration_subgroups=fig_calibration_subgroups, uq_models=fig_uq_models,
               selectivity=fig_selectivity)
    for t in todo:
        try:
            fns[t]()
        except FileNotFoundError as e:
            print("skip", t, "(missing)", e.filename)
    print("FIGURES_DONE")
