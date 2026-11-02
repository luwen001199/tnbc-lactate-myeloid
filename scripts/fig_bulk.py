#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_bulk.py -- publication figures for the bulk-tissue part of the study.
Figures 1-4 (+ supplementary). 300-600 dpi, Arial, colour-blind safe palette.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import mannwhitneyu, spearmanr

R = "/workspace/results"
F = "/workspace/figures"
os.makedirs(F, exist_ok=True)

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

PAL = {"Anti-tumour-trained": "#2166AC",       # blue
       "Intermediate": "#BDBDBD",              # grey
       "Pro-tumour-lactate-memory": "#B2182B"} # red
ORDER = ["Anti-tumour-trained", "Intermediate", "Pro-tumour-lactate-memory"]
SHORT = {"Anti-tumour-trained": "Anti-tumour trained",
         "Intermediate": "Intermediate",
         "Pro-tumour-lactate-memory": "Pro-tumour lactate-memory"}

tcga = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
_asg = pd.read_csv(f"{R}/subtype_assignments.csv", index_col=0)
for c in [x for x in _asg.columns if x.startswith("subtype_")]:
    tcga[c] = _asg[c].reindex(tcga.index)
met = pd.read_csv(f"{R}/metabric_scores.csv", index_col=0)
# Single source of truth for all Cox numbers in the figures: script 09.
# cox_m / cox_t are the covariate-adjusted models (METABRIC: age + nodes + NPI
# + chemo; TCGA-BRCA: age + AJCC stage) so that figure, Table 1 and Table S3
# report the same estimator.
_coxa = pd.read_csv(f"{R}/cox_multivariable.csv")
cox_m = _coxa[(_coxa.cohort == "METABRIC") & (_coxa.model == "adjusted")].copy()
cox_t = _coxa[(_coxa.cohort == "TCGA-BRCA") & (_coxa.model == "adjusted")].copy()

# ============================================================================
# Figure 1 -- conceptual roadmap
# ============================================================================
def fig1_roadmap():
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    boxes = [
        (2, "TNBC glycolytic\nmicroenvironment\n(hypoxia, LDHA, MCT1/4)", "#F4A582"),
        (21.5, "Lactate uptake by\nmonocytes / TAMs\n(SLC16A1, LDHA)", "#F4A582"),
        (41, "p300/CBP-dependent\nhistone lactylation\nH3K9la / H3K18la", "#D6604D"),
        (60.5, "Chromatin opening +\nhigh-CpG / low-methylation\nlate nt enhancer latch", "#B2182B"),
        (80, "Two myeloid states:\npro-tumour memory vs\nIRF8/CCL8 anti-tumour", "#4393C3"),
    ]
    for x, txt, c in boxes[:-1]:
        ax.add_patch(plt.Rectangle((x, 13), 15.5, 12, fc=c, ec="#333333", lw=0.7, alpha=0.95))
        ax.text(x + 7.75, 19, txt, ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", linespacing=1.5)
    x, txt, c = boxes[-1]
    ax.add_patch(plt.Rectangle((x, 13), 17, 12, fc=c, ec="#333333", lw=0.7, alpha=0.95))
    ax.text(x + 8.5, 19, txt, ha="center", va="center", fontsize=7, color="white",
            fontweight="bold", linespacing=1.5)

    for x0, x1 in [(17.5, 21.5), (37, 41), (56.5, 60.5), (76, 80)]:
        ax.annotate("", xy=(x1, 19), xytext=(x0, 19),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333333"))

    ax.text(50, 31, "Lactate-driven trained immunity in TNBC myeloid cells: hypothesis and analytic roadmap",
            ha="center", fontsize=9.5, fontweight="bold")

    # lower track: analytical layers
    layers = [("Bulk transcriptome\nTCGA + METABRIC", "#E8E8E8"),
              ("Single-cell myeloid\nGSE176078", "#E8E8E8"),
              ("Epigenetic latch\nATAC + 450K + CpG", "#E8E8E8"),
              ("Clinical translation\nsubtype, outcome, HDACi", "#E8E8E8")]
    for i, (t, c) in enumerate(layers):
        x = 2 + i * 24.5
        ax.add_patch(plt.Rectangle((x, 2.5), 20, 6.5, fc=c, ec="#666666", lw=0.6))
        ax.text(x + 10, 5.75, t, ha="center", va="center", fontsize=6.8, color="#333333",
                linespacing=1.4)
    ax.annotate("", xy=(98.5, 5.75), xytext=(22.5, 5.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#666666",
                                connectionstyle="arc3,rad=0"))
    ax.text(50, 10.2, "computational layers integrated in this study", ha="center",
            fontsize=7, style="italic", color="#555555")
    plt.savefig(f"{F}/Fig1_roadmap.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{F}/Fig1_roadmap.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Fig1 ok")


# ============================================================================
# Figure 2 -- the lactate axis is a TNBC-defining feature
# ============================================================================
def fig2_lactate_axis():
    fig = plt.figure(figsize=(9.2, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)

    # (a) lactate score TNBC vs non-TNBC in TCGA
    ax = fig.add_subplot(gs[0, 0])
    a = tcga.loc[tcga.is_tnbc, "lactate_core_adj"].dropna()
    b = tcga.loc[~tcga.is_tnbc, "lactate_core_adj"].dropna()
    u, p = mannwhitneyu(a, b)
    parts = ax.violinplot([b, a], showextrema=False, widths=0.75)
    for pc, c in zip(parts["bodies"], ["#D1E5F0", "#EF8A62"]):
        pc.set_facecolor(c); pc.set_alpha(0.85); pc.set_edgecolor("#333333"); pc.set_linewidth(0.6)
    for i, v in enumerate([b, a], start=1):
        ax.scatter(np.random.RandomState(i).normal(i, 0.045, len(v)), v, s=1.6,
                   c="#333333", alpha=0.25, linewidths=0)
        ax.hlines(v.median(), i - 0.22, i + 0.22, color="black", lw=1.4)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["non-TNBC", "TNBC"])
    ax.set_ylabel("Lactate-metabolism score\n(myeloid-adjusted, z)")
    ax.set_title("a  Lactate metabolism, TCGA-BRCA", loc="left", fontweight="bold")
    ax.text(0.5, 0.98, f"p = {p:.1e}", transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color="#B2182B", fontweight="bold")

    # (b) hypoxia scores across subtypes (TNBC)
    ax = fig.add_subplot(gs[0, 1])
    t = tcga[tcga.is_tnbc].dropna(subset=["subtype_TNBC"])
    for v in ["hypoxia_buffa", "hypoxia_winter"]:
        vals = [t.loc[t.subtype_TNBC == k, v].dropna() for k in ORDER]
        pos = np.arange(3) + (-0.18 if "buffa" in v else 0.18)
        ax.boxplot(vals, positions=pos, widths=0.3, patch_artist=True,
                   boxprops=dict(facecolor="#F4A582" if "buffa" in v else "#92C5DE",
                                 edgecolor="#333333", linewidth=0.5),
                   medianprops=dict(color="black", linewidth=0.9),
                   flierprops=dict(marker="o", ms=1.5, alpha=0.4, linewidth=0))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([SHORT[k] for k in ORDER], rotation=30, ha="right")
    ax.set_ylabel("Hypoxia score")
    ax.set_title("b  Hypoxia by myeloid subtype", loc="left", fontweight="bold")
    from scipy.stats import kruskal
    pv = kruskal(*[t.loc[t.subtype_TNBC == k, "hypoxia_buffa"].dropna() for k in ORDER])[1]
    ax.text(0.5, 0.98, f"Buffa p = {pv:.1e}", transform=ax.transAxes, ha="center",
            va="top", fontsize=7, color="#B2182B", fontweight="bold")

    # (c) subtype composition TNBC vs non-TNBC
    ax = fig.add_subplot(gs[0, 2])
    comp = pd.crosstab(tcga.subtype_ALLBRCA if "subtype_ALLBRCA" in tcga else tcga["subtype_ALL-BRCA"],
                       tcga.is_tnbc, normalize="columns").T.reindex(columns=ORDER) * 100
    comp.plot(kind="bar", stacked=True, ax=ax, color=[PAL[k] for k in ORDER],
              edgecolor="white", width=0.55, linewidth=0.5)
    ax.set_ylabel("% of samples"); ax.set_xticklabels(["non-TNBC", "TNBC"], rotation=0)
    ax.set_xlabel(""); ax.legend_.remove()
    ax.set_title("c  Myeloid subtype composition", loc="left", fontweight="bold")
    for i, (idx, row) in enumerate(comp.iterrows()):
        cum = 0
        for k in ORDER:
            v = row[k]
            if v > 6:
                ax.text(i, cum + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
            cum += v

    # (d) scatter LT_pro vs LT_anti, TCGA-TNBC
    ax = fig.add_subplot(gs[1, 0])
    for k in ORDER:
        s = t[t.subtype_TNBC == k]
        ax.scatter(s.LT_pro, s.LT_anti, s=9, c=PAL[k], alpha=0.8, edgecolors="none",
                   label=SHORT[k])
    ax.axhline(0, color="#888888", lw=0.6, ls="--"); ax.axvline(0, color="#888888", lw=0.6, ls="--")
    ax.set_xlabel("Pro-tumour lactate-memory score")
    ax.set_ylabel("Anti-tumour trained score")
    r = spearmanr(t.LT_pro, t.LT_anti)
    ax.set_title(f"d  Dual axes (rho={r.correlation:+.2f})", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")

    # (e) immune contexture heatmap
    ax = fig.add_subplot(gs[1, 1:])
    feats = ["mcp_CD8_T", "mcp_T_cell", "mcp_NK", "mcp_B_cell", "mcp_Myeloid_DC",
             "mcp_Monocytic", "mcp_Neutrophil", "trained_anti_adj", "hdaci_retrain_adj",
             "innate_antiviral_adj", "lactate_core_adj", "lactate_metab_adj",
             "trained_pro_adj", "hypoxia_buffa"]
    lab = ["CD8 T", "T cell", "NK", "B cell", "myeloid DC", "monocytic", "neutrophil",
           "trained-anti", "HDACi-retrain", "antiviral ISG", "lactate-core",
           "lactate-metab", "trained-pro", "hypoxia"]
    hm = t.groupby("subtype_TNBC")[feats].mean().reindex(ORDER)
    hmz = hm.sub(hm.mean(axis=0), axis=1).div(hm.std(axis=0) + 1e-9, axis=1)
    sns.heatmap(hmz.T, cmap="RdBu_r", center=0, vmin=-2.2, vmax=2.2, ax=ax,
                cbar_kws=dict(label="row z-score", shrink=0.6), linewidths=0.4,
                linecolor="white", xticklabels=[SHORT[k] for k in ORDER])
    ax.set_yticklabels(lab, rotation=0); ax.set_xlabel("")
    ax.set_title("e  Immune and metabolic contexture by subtype (TCGA-TNBC)",
                 loc="left", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.savefig(f"{F}/Fig2_lactate_axis.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{F}/Fig2_lactate_axis.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Fig2 ok")


# ============================================================================
# Figure 3 -- TCGA survival
# ============================================================================
def km_panel(ax, df, subcol, ep, title, atrisk=True, tmax=None):
    d = df.dropna(subset=[f"{ep}_months", f"{ep}_event", subcol])
    d = d[d[f"{ep}_months"] > 0]
    fitted = {}
    for k in ORDER:
        g = d[d[subcol] == k]
        if len(g) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(g[f"{ep}_months"], g[f"{ep}_event"], label=f"{SHORT[k]} (n={len(g)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=PAL[k], lw=1.5)
        fitted[k] = len(g)
    lr = multivariate_logrank_test(d[f"{ep}_months"], d[subcol], d[f"{ep}_event"])
    ax.set_xlabel("")
    ax.text(0.5, -0.40, "Months", transform=ax.transAxes,
            ha="center", va="top", fontsize=7)
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    tlim = tmax or min(300, float(d[f"{ep}_months"].max()) * 1.01)
    ax.set_xlim(0, tlim)
    ax.set_title(f"{title}\nlog-rank p = {lr.p_value:.2e}", loc="left",
                 fontweight="bold", fontsize=8)
    ax.legend(frameon=False, fontsize=6.3, loc=(0.02, 0.02))

    if atrisk and fitted:
        tps = np.linspace(0, tlim * 0.85, 5).round(0)
        xcol = np.linspace(0.44, 0.97, len(tps))
        ax.text(-0.10, -0.115, "No. at risk", transform=ax.transAxes, fontsize=5.3,
                fontweight="bold", clip_on=False)
        for j, t in enumerate(tps):
            ax.text(xcol[j], -0.115, f"{int(t)}", transform=ax.transAxes, fontsize=5.3,
                    ha="center", clip_on=False)
        for i, k in enumerate([k for k in ORDER if k in fitted]):
            y = -0.175 - i * 0.062
            ax.text(-0.10, y, SHORT[k], transform=ax.transAxes, fontsize=5.3,
                    color=PAL[k], clip_on=False)
            for j, t in enumerate(tps):
                n_at = int(((d[subcol] == k) & (d[f"{ep}_months"] >= t)).sum())
                ax.text(xcol[j], y, str(n_at), transform=ax.transAxes, fontsize=5.3,
                        ha="center", clip_on=False)
    return lr.p_value


def fig3_tcga_survival():
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1))
    km_panel(axes[0], tcga, "subtype_ALL-BRCA", "os", "a  Overall survival, TCGA-BRCA")
    km_panel(axes[1], tcga, "subtype_ALL-BRCA", "dss", "b  Disease-specific survival")
    t = tcga[tcga.is_tnbc]
    km_panel(axes[2], t, "subtype_TNBC", "os", "c  Overall survival, TCGA-TNBC")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.savefig(f"{F}/Fig3_tcga_survival.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{F}/Fig3_tcga_survival.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Fig3 ok")


# ============================================================================
# Figure 4 -- METABRIC validation
# ============================================================================
def _forest(ax, rows, xlabel, title, sig_col="#B2182B", ns_col="#999999"):
    """rows: list of dict(label, hr, lo, hi, p, note, sig)"""
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        col = sig_col if r["sig"] else ns_col
        mk = "s" if r["sig"] else "o"
        ax.errorbar(np.log2(r["hr"]), i,
                    xerr=[[np.log2(r["hr"]) - np.log2(r["lo"])],
                          [np.log2(r["hi"]) - np.log2(r["hr"])]],
                    fmt=mk, ms=5 if r["sig"] else 4.5, color=col, ecolor=col,
                    elinewidth=1.3, capsize=2.5,
                    mfc=col if r["sig"] else "white", mec=col)
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ticks = [0.5, 1, 2, 4]
    ax.set_xticks([np.log2(t) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    lo = min(np.log2(r["lo"]) for r in rows) - 0.35
    has_note = any(r.get("note") for r in rows)
    hi = max(np.log2(r["hi"]) for r in rows) + (2.55 if has_note else 0.75)
    ax.set_xlim(lo, hi)
    for i, r in enumerate(rows):
        ptxt = f"{r['p']:.1e}" if r["p"] < 1e-3 else f"{r['p']:.3f}"
        ax.text(hi - (1.55 if has_note else 0.05), i, ptxt, fontsize=5.6,
                va="center", ha="right")
        if r.get("note"):
            ax.text(hi - 0.03, i, r["note"], fontsize=5.2, va="center", ha="right",
                    color="#555555")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8)


def fig4_metabric():
    fig = plt.figure(figsize=(9.6, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.36,
                          top=0.93, bottom=0.08, left=0.075, right=0.975)
    st = met[met.is_tnbc.fillna(False)]

    # (a,b) KM -- METABRIC TNBC
    km_panel(fig.add_subplot(gs[0, 0]), st, "subtype", "os",
             "a  OS, METABRIC TNBC (n=286)")
    km_panel(fig.add_subplot(gs[0, 1]), st, "subtype", "rfs",
             "b  RFS, METABRIC TNBC (n=285)")

    # (c) forest of continuous HRs, METABRIC TNBC OS, with BH-FDR
    ax = fig.add_subplot(gs[0, 2])
    d = cox_m[(cox_m.scope == "TNBC") & (cox_m.endpoint == "os")].copy()
    d = d[d["var"].isin(["LT_pro", "LT_anti", "lactate_core_adj",
                         "trained_anti_adj", "trained_pro_adj"])].copy()
    d = d.sort_values("hr_sd")
    d["fdr"] = np.minimum(1, d.p * len(d))
    rows = [dict(label=v.replace("_adj", "").replace("_", "-"), hr=r.hr_sd, lo=r.lo,
                 hi=r.hi, p=r.fdr, sig=r.fdr < 0.05, note="")
            for v, r in d.set_index("var").iterrows()]
    _forest(ax, rows, "HR per 1 SD (log scale)",
            "c  METABRIC TNBC, OS, adjusted\n(n=279, 150 events; BH-FDR)", sig_col="#2166AC")
    ax.text(0.02, -0.14, "p = BH-adjusted", transform=ax.transAxes, fontsize=5.2,
            color="#555555")

    # (d,e) KM -- METABRIC all subtypes
    km_panel(fig.add_subplot(gs[1, 0]), met, "subtype", "os",
             "d  OS, METABRIC all subtypes (n=1,979)")
    km_panel(fig.add_subplot(gs[1, 1]), met, "subtype", "rfs",
             "e  RFS, METABRIC all subtypes (n=1,976)")

    # (f) cross-cohort consistency -- every row annotated with n / events;
    #     TCGA OS rows flagged: all-cause deaths dilute the signal
    def g(scope, ep, var, src):
        t = (cox_m if src == "m" else cox_t)
        s = t[(t.scope == scope) & (t.endpoint == ep) & (t["var"] == var)]
        r = s.iloc[0]
        return dict(label=f"{scope} · {ep.upper()}", hr=r.hr_sd, lo=r.lo, hi=r.hi,
                    p=r.p, sig=r.p < 0.05, note=f"n={int(r.n)}, ev={int(r.events)}")

    rows = [dict(g("TNBC", "os", "LT_pro", "m"), label="METABRIC TNBC · OS"),
            dict(g("ALL", "os", "LT_pro", "m"), label="METABRIC all · OS"),
            dict(g("ALL-BRCA", "dss", "LT_pro", "t"), label="TCGA-BRCA · DSS"),
            dict(g("ALL-BRCA", "os", "LT_pro", "t"), label="TCGA-BRCA · OS*"),
            dict(g("TNBC", "os", "LT_pro", "t"), label="TCGA TNBC · OS*")]
    ax = fig.add_subplot(gs[1, 2])
    _forest(ax, rows, "HR per 1 SD (log scale)",
            "f  Cross-cohort consistency, LT_pro")
    ax.text(0.02, -0.20, "* TCGA-BRCA OS records all-cause deaths and is null here\n"
                         "  (HR 1.24, 0.80-1.92, p=0.34); DSS carries the signal.",
            transform=ax.transAxes, fontsize=5.0, color="#777777", va="top")

    for a in fig.axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    plt.savefig(f"{F}/Fig4_metabric_validation.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{F}/Fig4_metabric_validation.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Fig4 ok")


# ============================================================================
# Supplementary: subtype-independent prognostic value & correlation heatmap
# ============================================================================
def figS_scores():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    # (a) myeloid-abundance confounding
    ax = axes[0]
    for c, lab, col in [("trained_anti", "trained-anti (raw)", "#B2182B"),
                        ("trained_pro", "trained-pro (raw)", "#2166AC")]:
        ax.scatter(tcga.myeloid_abundance, tcga[c], s=3, alpha=0.25, c=col, label=lab,
                   edgecolors="none")
    for c in ["trained_anti", "trained_pro"]:
        b = np.polyfit(tcga.myeloid_abundance, tcga[c], 1)
        xs = np.linspace(tcga.myeloid_abundance.min(), tcga.myeloid_abundance.max(), 50)
        ax.plot(xs, np.polyval(b, xs), lw=1.2, color="#333333")
    ax.set_xlabel("Myeloid abundance (curated lineage markers)")
    ax.set_ylabel("Signature score")
    r1 = np.corrcoef(tcga.myeloid_abundance, tcga.trained_anti)[0, 1]
    ax.set_title(f"a  Raw scores track infiltrate (r={r1:.2f})",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.2)

    # (b) after residualisation
    ax = axes[1]
    r2 = np.corrcoef(tcga.myeloid_abundance, tcga.trained_anti_adj)[0, 1]
    ax.scatter(tcga.myeloid_abundance, tcga.trained_anti_adj, s=3, alpha=0.25,
               c="#2166AC", edgecolors="none")
    ax.axhline(0, color="#888888", lw=0.6, ls="--")
    ax.set_xlabel("Myeloid abundance"); ax.set_ylabel("trained-anti (adjusted)")
    ax.set_title(f"b  After residualisation (r={r2:.2f})", loc="left",
                 fontweight="bold", fontsize=8.5)

    # (c) distribution of the two axes by subtype (TNBC)
    ax = axes[2]
    t = tcga[tcga.is_tnbc].dropna(subset=["subtype_TNBC"])
    dat = [t.loc[t.subtype_TNBC == k, "LT_pro"] for k in ORDER]
    ax.boxplot(dat, positions=np.arange(3) - 0.18, widths=0.3, patch_artist=True,
               boxprops=dict(facecolor="#EF8A62", edgecolor="#333333", linewidth=0.5),
               medianprops=dict(color="black", linewidth=0.9),
               flierprops=dict(marker="o", ms=1.5, alpha=0.4, linewidth=0))
    dat2 = [t.loc[t.subtype_TNBC == k, "LT_anti"] for k in ORDER]
    ax.boxplot(dat2, positions=np.arange(3) + 0.18, widths=0.3, patch_artist=True,
               boxprops=dict(facecolor="#67A9CF", edgecolor="#333333", linewidth=0.5),
               medianprops=dict(color="black", linewidth=0.9),
               flierprops=dict(marker="o", ms=1.5, alpha=0.4, linewidth=0))
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([SHORT[k] for k in ORDER],
                                                    rotation=30, ha="right")
    ax.set_ylabel("Score (z)")
    ax.set_title("c  Dual scores per subtype (TNBC)", loc="left",
                 fontweight="bold", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#EF8A62", label="LT-pro"),
                       Patch(facecolor="#67A9CF", label="LT-anti")],
              frameon=False, fontsize=6.2)
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.savefig(f"{F}/FigS1_score_construction.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(f"{F}/FigS1_score_construction.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("FigS1 ok")


if __name__ == "__main__":
    fig1_roadmap()
    fig2_lactate_axis()
    fig3_tcga_survival()
    fig4_metabric()
    figS_scores()
    print("ALL bulk figures done")
