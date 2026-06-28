import json
import pandas as pd
from plotnine import (ggplot, aes, geom_line, geom_ribbon, geom_point, geom_text,
                      geom_abline, geom_hline, facet_wrap, theme_bw, theme, labs,
                      scale_linetype_manual, scale_shape_manual, element_text,
                      element_rect, guides, guide_legend)

ORDER = ["random", "greedy", "uncertainty", "ucb", "ei", "hybrid"]
LINETYPES = {"random":"solid","greedy":"dashed","uncertainty":"dotted",
             "ucb":"dashdot","ei":(0, (6, 2)),"hybrid":(0,(3,1,1,1,1,1))}
SHAPES = {"random":"o","greedy":"s","uncertainty":"^","ucb":"D","ei":"v","hybrid":"P"}
LT = ["solid","dashed","dotted","dashdot",(0, (6, 2)),(0,(3,1,1,1,1,1))]
SH = ["o","s","^","D","v","P"]

def long_df(path):
    d = json.load(open(path))["results"]; rows = []
    labels = {"r2":"Prediction: test R²", "recall":"Design: top-10 recall"}
    for s in ORDER:
        if s not in d: continue
        r = d[s]
        for i, x in enumerate(r["n_queried"]):
            for key in ("r2", "recall"):
                m = r[f"{key}_mean"][i]; sd = r[f"{key}_std"][i]
                rows.append(dict(strategy=s, x=x, metric=labels[key],
                                 value=m, lo=m-sd, hi=m+sd))
    df = pd.DataFrame(rows)
    df["strategy"] = pd.Categorical(df["strategy"], categories=ORDER, ordered=True)
    return df, json.load(open(path))

def gg_curves(path, out, title):
    df, raw = long_df(path)
    p = (ggplot(df, aes("x", "value"))
         + geom_ribbon(aes(ymin="lo", ymax="hi", group="strategy"),
                       fill="#B3B3B3", alpha=0.25)
         + geom_line(aes(linetype="strategy"), size=0.7)
         + geom_point(aes(shape="strategy"), size=1.4, data=df[df.x % 90 == df.x.min() % 90])
         + facet_wrap("metric", scales="free_y")
         + scale_linetype_manual(values=LT)
         + scale_shape_manual(values=SH)
         + labs(x="number of experimental measurements", y="", linetype="", shape="")
         + theme_bw(base_size=11)
         + theme(figure_size=(8.2, 3.4), legend_position="right",
                 strip_background=element_rect(fill="#E6E6E6"),
                 strip_text=element_text(weight="bold")))
    p.save(out, dpi=300, verbose=False)
    print("wrote", out)

def gg_calibration(out):
    d = json.load(open("results/calibration.json"))["coverage"]
    noms = [0.5, 0.8, 0.9]; rows = []
    for nom in noms:
        rows.append(dict(nominal=nom, empirical=d[str(nom)]["uncalibrated"], method="uncalibrated"))
        rows.append(dict(nominal=nom, empirical=d[str(nom)]["conformal"], method="conformal"))
    df = pd.DataFrame(rows)
    p = (ggplot(df, aes("nominal", "empirical"))
         + geom_abline(intercept=0, slope=1, linetype="solid", color="#8C8C8C", size=0.5)
         + geom_line(aes(linetype="method"), size=0.7)
         + geom_point(aes(shape="method"), size=2.6, fill="white")
         + scale_linetype_manual(values=["dashed", "dotted"])
         + scale_shape_manual(values=["s", "o"])
         + labs(x="nominal coverage", y="empirical coverage", linetype="", shape="")
         + theme_bw(base_size=11)
         + theme(figure_size=(4.0, 3.4), legend_position="right"))
    p.save(out, dpi=300, verbose=False)
    print("wrote", out)

def gg_toc(out):
    d = json.load(open("results/real_al_results.json"))["results"]
    rows = [dict(strategy=s, recall=d[s]["recall_mean"][-1], r2=d[s]["r2_mean"][-1]) for s in ORDER]
    df = pd.DataFrame(rows)
    df["strategy"] = pd.Categorical(df["strategy"], categories=ORDER, ordered=True)
    p = (ggplot(df, aes("recall", "r2"))
         + geom_point(aes(shape="strategy"), size=4, fill="white", stroke=0.8)
         + geom_text(aes(label="strategy"), size=7, nudge_y=0.012, va="bottom")
         + scale_shape_manual(values=SH)
         + labs(x="design: top-10 recall", y="prediction: R²",
                title="No acquisition wins both objectives", shape="")
         + theme_bw(base_size=10)
         + theme(figure_size=(3.3, 2.1), legend_position="none",
                 plot_title=element_text(weight="bold", size=9)))
    p.save(out, dpi=300, verbose=False)
    print("wrote", out)

if __name__ == "__main__":
    gg_curves("results/real_al_results.json",
              "figures/real_al.pdf", "f-element extraction")
    gg_curves("results/lipo_al_results.json",
              "figures/lipo_al.pdf", "Lipophilicity")
    gg_calibration("figures/calibration.pdf")
    gg_toc("paper_b/TOC_graphic.png")
    gg_toc("paper_b/TOC_graphic.tiff")
    print("GGPLOT_FIGURES_DONE")
