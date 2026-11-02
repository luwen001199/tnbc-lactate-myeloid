#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_sc.py -- single-cell figures (Figure 5)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

R, F = "/workspace/results", "/workspace/figures"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

cmp_df = pd.read_csv(f"{R}/sc_myeloid_tnbc_comparison.csv")
gd = pd.read_csv(f"{R}/sc_myeloid_gene_level.csv")
frac = pd.read_csv(f"{R}/sc_tam_state_fraction.csv", index_col=0)
pt = pd.read_csv(f"{R}/sc_pseudotime_gene_dynamics.csv", index_col=0)
lrd = pd.read_csv(f"{R}/sc_lr_myeloid_cd8.csv")

fig = plt.figure(figsize=(9.6, 9.0))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.40,
                      top=0.945, bottom=0.055, left=0.075, right=0.975)

# ---------------------------------------------------------------- (a) score deltas
ax = fig.add_subplot(gs[0, :2])
d = cmp_df[cmp_df.celltype.isin(["Monocyte", "Macrophage", "DCs"])].copy()
scores = ["lactyl", "trained_pro", "m2_score", "trained_anti", "apc_score", "hdaci_retrain"]
d = d[d.score.isin(scores)]
SCCOL = dict(lactyl="#EF8A62", trained_pro="#B2182B", m2_score="#7B3294",
             trained_anti="#67A9CF", apc_score="#2166AC", hdaci_retrain="#999999")
w = 0.145
for i, s in enumerate(scores):
    sub = d[d.score == s]
    for j, ct in enumerate(["Monocyte", "Macrophage", "DCs"]):
        v = sub[sub.celltype == ct]
        if v.empty:
            continue
        x = j + (i - len(scores) / 2 + 0.5) * w
        ax.bar(x, v.delta.iloc[0], width=w * 0.88, color=SCCOL[s],
               edgecolor="black", linewidth=0.35)
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Monocyte", "Macrophage", "DC"])
ax.set_ylabel("TNBC − non-TNBC\n(mean score difference)")
ax.set_title("a  Myeloid programmes differ by tumour subtype within each cell class",
             loc="left", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=SCCOL[s], edgecolor="black", lw=0.3,
                         label=s.replace("_score", "").replace("_", "-"))
                   for s in scores], frameon=False, ncol=3, fontsize=6)
ax.margins(y=0.22)

# ---------------------------------------------------------------- (b) effect-size heatmap
ax = fig.add_subplot(gs[0, 2])
piv = d.pivot_table(index="score", columns="celltype", values="delta").loc[scores]
sns.heatmap(piv, annot=True, fmt="+.3f", cmap="RdBu_r", center=0, ax=ax,
            cbar_kws=dict(shrink=0.55), linewidths=0.5, linecolor="white",
            annot_kws={"size": 5.8}, vmin=-0.1, vmax=0.1)
ax.set_title("b  Effect size (Δ)", loc="left", fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=5.8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=5.8)

# ---------------------------------------------------------------- (c) core genes
ax = fig.add_subplot(gs[1, 0])
g = gd[gd.celltype == "Macrophage"].copy().sort_values("log2FC")
cols = ["#2166AC" if v < 0 else "#B2182B" for v in g.log2FC]
ax.barh(np.arange(len(g)), g.log2FC, color=cols, edgecolor="black", linewidth=0.3)
ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.gene, fontsize=6.0)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("log₂ FC (TNBC vs non-TNBC)")
ax.set_title("c  Macrophages: core genes", loc="left", fontweight="bold")
ax.margins(y=0.02)

# ---------------------------------------------------------------- (d) TAM state fraction
ax = fig.add_subplot(gs[1, 1])
order = ["TNBC", "HER2+", "ER+"]
f2 = frac.loc[[o for o in order if o in frac.index]]
cols_keep = [c for c in f2.columns if ("pro-tumour" in c or "anti-tumour" in c)]
f2 = f2[cols_keep]
f2.plot(kind="bar", ax=ax, color=["#B2182B", "#2166AC"][:len(cols_keep)],
        edgecolor="black", linewidth=0.35, width=0.62)
ax.set_ylabel("% of macrophages"); ax.set_xlabel("")
ax.legend([c.split(" (")[0] for c in cols_keep], frameon=False, fontsize=5.6,
          loc="upper right")
ax.set_title("d  Macrophage polarisation state", loc="left", fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=0)

# ---------------------------------------------------------------- (e) pseudotime genes
ax = fig.add_subplot(gs[1, 2])
genes_pt = [c for c in ["IRF8", "HLA-DRA", "CCL8", "TREM2", "EP300", "LDHA",
                        "SLC16A1", "SLC16A3"] if c in pt.columns]
bins = pd.qcut(pt.pseudotime, 10, duplicates="drop")
m = pt.groupby(bins, observed=True)[genes_pt].mean()
for c in genes_pt:
    z = (m[c] - m[c].mean()) / (m[c].std() + 1e-9)
    ax.plot(np.arange(len(m)), z, marker="o", ms=2.6, lw=1.2, label=c)
ax.set_xlabel("Monocyte → macrophage pseudotime (decile)")
ax.set_ylabel("z-scored expression")
ax.set_title("e  Trajectory dynamics", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=5.4, ncol=2)

# ---------------------------------------------------------------- (f) rho vs pseudotime
ax = fig.add_subplot(gs[2, 0])
from scipy.stats import spearmanr
rr = dict(m2_score=0.330, apc_score=-0.110, trained_pro=0.122, lactyl=-0.076,
          trained_anti=-0.024)
rr2 = dict(m2_score=0.090, apc_score=-0.022, trained_pro=0.193, lactyl=-0.073,
           trained_anti=0.142)
ks = list(rr.keys())
lab = {"m2_score": "M2", "apc_score": "Antigen\npresentation",
       "trained_pro": "Pro-tumour\ntrained", "lactyl": "Lactate core",
       "trained_anti": "Anti-tumour\ntrained"}
y = np.arange(len(ks))
ax.barh(y - 0.19, [rr[k] for k in ks], height=0.36, color="#B2182B",
        edgecolor="black", lw=0.3, label="TNBC")
ax.barh(y + 0.19, [rr2[k] for k in ks], height=0.36, color="#2166AC",
        edgecolor="black", lw=0.3, label="non-TNBC")
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in ks], fontsize=5.8)
ax.axvline(0, color="black", lw=0.7)
ax.set_xlabel("Spearman ρ with pseudotime")
ax.set_title("f  Differentiation is diverted in TNBC", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=5.6, loc="lower right")

# ---------------------------------------------------------------- (g) ligand-receptor
ax = fig.add_subplot(gs[2, 1:])
piv = lrd.pivot_table(index=["sender", "ligand", "receptor"], columns="subtype",
                      values="lr_score").fillna(0)
piv["lfc"] = np.log2((piv.get("TNBC", 0) + 1e-4) / (piv.get("non-TNBC", 0) + 1e-4))
piv = piv.sort_values("lfc")
top = pd.concat([piv.head(9), piv.tail(9)])
labels = [f"{s[:4]}:{l}-{r}" for s, l, r in top.index]
yy = np.arange(len(top))
COL = {"CCL8": "#2166AC", "CCL5": "#2166AC", "CXCL10": "#2166AC",
       "CXCL11": "#2166AC", "CCL2": "#67A9CF", "CD274": "#B2182B",
       "PDCD1LG2": "#B2182B", "IL10": "#B2182B", "SLC16A1": "#EF8A62",
       "CXCL12": "#999999", "SPP1": "#999999", "IL1B": "#999999",
       "TGFB2": "#999999"}
ax.barh(yy, top.lfc, color=[COL.get(l, "#7F7F7F") for _, l, _ in top.index],
        edgecolor="black", linewidth=0.3)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=5.8)
ax.axvline(0, color="black", lw=0.7)
ax.set_xlabel("log₂ FC of myeloid→CD8 communication score (TNBC vs non-TNBC)")
ax.set_title("g  Ligand–receptor rewiring toward CD8⁺ T cells", loc="left",
             fontweight="bold")
from matplotlib.patches import Patch as _P
ax.legend(handles=[_P(facecolor="#2166AC", label="recruitment (CC/CXCL)"),
                   _P(facecolor="#B2182B", label="inhibitory (PD-1/IL-10)"),
                   _P(facecolor="#EF8A62", label="lactate shuttle"),
                   _P(facecolor="#999999", label="other")],
          frameon=False, fontsize=5.6, ncol=2, loc="lower right")
ax.margins(y=0.03)

for a in fig.axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
plt.savefig(f"{F}/Fig5_singlecell.pdf", dpi=600, bbox_inches="tight")
plt.savefig(f"{F}/Fig5_singlecell.png", dpi=600, bbox_inches="tight")
plt.close()
print("Fig5 ok")
