#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_epi.py -- chromatin-layer figure (Figure 6)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

R, F = "/workspace/results", "/workspace/figures"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

arch = pd.read_csv(f"{R}/epi_promoter_architecture_panels.csv")
enr = pd.read_csv(f"{R}/epi_architecture_enrichment.csv")
md = pd.read_csv(f"{R}/epi_methyl_by_state_TNBC.csv")
prog = pd.read_csv(f"{R}/epi_programme_methylation_tnbc.csv")
me = pd.read_csv(f"{R}/epi_methylation_expression.csv")
lat = pd.read_csv(f"{R}/epi_latched_by_programme.csv", index_col=0)
GB = pd.read_csv(f"{R}/meth_gene_promoter.csv", index_col=0)
sco = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)

PROG_ORDER = ["Anti-tumour trained", "Pro-tumour suppressive", "Lactate metabolism",
              "Lactylation machinery", "Innate antiviral (control)"]
PCOL = {"Anti-tumour trained": "#2166AC", "Pro-tumour suppressive": "#B2182B",
        "Lactate metabolism": "#EF8A62", "Lactylation machinery": "#762A83",
        "Innate antiviral (control)": "#7F7F7F"}

fig = plt.figure(figsize=(9.6, 8.4))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.38,
                      top=0.93, bottom=0.07, left=0.075, right=0.975)

# ---------------------------------------------------------------- (a) architecture heatmap
ax = fig.add_subplot(gs[0, 0])
A = arch.set_index("programme").loc[PROG_ORDER]
cols = [("monoA_k27ac_pct", "Monocyte\nH3K27ac"), ("monoA_k4me3_pct", "Monocyte\nH3K4me3"),
        ("monoB_k27ac_pct", "Monocyte 2\nH3K27ac"), ("cpg_island_pct", "CpG\nisland"),
        ("open_promoter_pct", "DHS\nopen")]
M = A[[c for c, _ in cols]]
M.columns = [l for _, l in cols]
sns.heatmap(M, annot=True, fmt=".0f", cmap="YlOrRd", vmin=0, vmax=100, ax=ax,
            cbar_kws=dict(shrink=0.6, label="% of promoters"),
            linewidths=0.5, linecolor="white", annot_kws={"size": 6})
ax.set_yticklabels([p.replace(" ", "\n") for p in PROG_ORDER], rotation=0, fontsize=5.8)
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=5.6)
ax.set_title("a  Promoter chromatin architecture", loc="left", fontweight="bold")

# ---------------------------------------------------------------- (b) enrichment forest
ax = fig.add_subplot(gs[0, 1:])
e = enr[(enr.set == "panel") & (enr.feature.isin(["monoA_k27ac", "monoA_k4me3",
                                                  "monoB_k27ac", "monoB_k4me3"]))].copy()
e["prog"] = pd.Categorical(e.programme, PROG_ORDER, ordered=True)
e = e.sort_values(["feature", "prog"])
y = np.arange(len(e))
ax.scatter(np.log2(e.OR.clip(lower=0.05)), y, s=18,
           c=[PCOL[p] for p in e.programme], edgecolors="none")
ax.axvline(0, color="#777777", ls="--", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([f"{r.programme.replace(' (control)','')} · {r.feature.replace('mono','').replace('_','/')}"
                    for _, r in e.iterrows()], fontsize=5.3)
ax.set_xlim(-4.6, 4.0)
ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_xticklabels(["0.06", "0.25", "1", "4", "16"])
ax.set_xlabel("Odds ratio vs all RefSeq promoters (log2)")
ax.set_title("b  Myeloid active-promoter enrichment", loc="left", fontweight="bold")
for i, (_, r) in enumerate(e.iterrows()):
    ax.text(3.55, i, "*" if r.fdr < 0.05 else "n.s.", fontsize=5.2, va="center", ha="right")
ax.text(0.0, 1.02, "all rows FDR < 0.05 unless marked", transform=ax.transAxes,
        fontsize=5.0, color="#777777")

# ---------------------------------------------------------------- (c) per-patient dBeta
ax = fig.add_subplot(gs[1, 0])
KEY = ["IRF8", "CIITA", "NLRC5", "HLA-B", "B2M", "CD74", "HLA-DRA",
       "LDHA", "SLC16A1", "ENO1"]
tn = sco[sco.is_tnbc == True].index if "is_tnbc" in sco.columns else sco.index
common = [s for s in GB.columns if s in tn]
S = sco.loc[common]
lo_pro, hi_pro = S.LT_pro.quantile(0.25), S.LT_pro.quantile(0.75)
lo_an, hi_an = S.LT_anti.quantile(0.25), S.LT_anti.quantile(0.75)
hp = S.index[S.LT_pro > hi_pro]
ha = S.index[S.LT_anti > hi_an]
rows = []
for g in KEY:
    if g not in GB.index:
        continue
    v = GB.loc[g, common]
    a, b = v[hp].dropna(), v[ha].dropna()
    if len(a) < 10 or len(b) < 10:
        continue
    _, p = mannwhitneyu(a, b)
    rows.append(dict(gene=g, d=a.mean() - b.mean(), p=p))
dd = pd.DataFrame(rows).sort_values("d")
dd["grp"] = ["APC / IRF8" if g in ["IRF8", "CIITA", "NLRC5", "HLA-B", "B2M", "CD74", "HLA-DRA"]
             else "Lactate" for g in dd.gene]
cols = ["#2166AC" if g == "APC / IRF8" else "#EF8A62" for g in dd.grp]
ax.barh(np.arange(len(dd)), dd.d, color=cols, edgecolor="black", linewidth=0.35)
ax.set_yticks(np.arange(len(dd))); ax.set_yticklabels(dd.gene, fontsize=6.2)
ax.axvline(0, color="black", lw=0.7)
for i, (_, r) in enumerate(dd.iterrows()):
    ax.text(r.d + (0.002 if r.d >= 0 else -0.002), i,
            "***" if r.p < 1e-3 else ("**" if r.p < 0.01 else ("*" if r.p < 0.05 else "n.s.")),
            fontsize=5.4, va="center", ha="left" if r.d >= 0 else "right")
ax.set_xlabel(u"Δβ  (LT-pro high − LT-anti high)")
ax.set_title("c  Per-tumour promoter methylation\n(TNBC, extreme quartiles)",
             loc="left", fontweight="bold")
ax.margins(x=0.22)

# ---------------------------------------------------------------- (d) programme level
ax = fig.add_subplot(gs[1, 1])
pl = prog.set_index("programme").loc[[p for p in PROG_ORDER if p in set(prog.programme)]]
y = np.arange(len(pl))
ax.barh(y, pl.delta, color=[PCOL[p] for p in pl.index], edgecolor="black", linewidth=0.35)
ax.set_yticks(y)
ax.set_yticklabels([p.replace(" ", "\n").replace("(control)", "(ctrl)") for p in pl.index],
                   fontsize=5.6)
ax.axvline(0, color="black", lw=0.7)
for i, (_, r) in enumerate(pl.iterrows()):
    ax.text(r.delta + 0.0025 if r.delta >= 0 else r.delta - 0.0025, i,
            f"p={r.p:.3f}", fontsize=5.3, va="center",
            ha="left" if r.delta >= 0 else "right")
ax.set_xlabel(u"Δβ (LT-pro high − LT-anti high)")
ax.set_title("d  Programme-level, TNBC", loc="left", fontweight="bold")
ax.margins(x=0.32)

# ---------------------------------------------------------------- (e) methylation-expression
ax = fig.add_subplot(gs[1, 2])
piv = me.groupby("programme").rho.apply(list)
order = [p for p in PROG_ORDER if p in piv.index]
bp = ax.boxplot([piv[p] for p in order], vert=True, widths=0.55, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", lw=1.0))
for b, p in zip(bp["boxes"], order):
    b.set_facecolor(PCOL[p]); b.set_edgecolor("black"); b.set_linewidth(0.4)
ax.axhline(0, color="#777777", ls="--", lw=0.7)
ax.set_xticklabels([p.replace(" ", "\n").replace("(control)", "(ctrl)") for p in order],
                   fontsize=5.2)
ax.set_ylabel("Spearman ρ (promoter β vs expression)")
ax.set_title("e  Methylation–expression\ncoupling", loc="left", fontweight="bold")

# ---------------------------------------------------------------- (f) latched genes
ax = fig.add_subplot(gs[2, :2])
L = lat.copy()
L["pct"] = 100 * L.latched / L.n
L = L.loc[[p for p in PROG_ORDER if p in L.index]]
y = np.arange(len(L))
ax.barh(y, L.pct, color=[PCOL[p] for p in L.index], edgecolor="black", linewidth=0.35)
ax.set_yticks(y)
ax.set_yticklabels([p.replace(" ", "\n").replace("(control)", "(ctrl)") for p in L.index],
                   fontsize=5.6)
for i, (idx, r) in enumerate(L.iterrows()):
    ax.text(r.pct + 1.2, i, f"{int(r.latched)}/{int(r.n)}", fontsize=5.6, va="center")
ax.set_xlabel("% of panel genes that are CpG-island + open + hypomethylated (β<0.3) + ρ<−0.1")
ax.set_title("f  'Latched' chromatin configuration by programme", loc="left",
             fontweight="bold")
ax.margins(x=0.18)

# ---------------------------------------------------------------- (g) exemplar scatter
ax = fig.add_subplot(gs[2, 2])
expr = pd.read_pickle(f"{R}/expr_tcga_brca.pkl")
sh = [s for s in common if s in expr.columns]
g = "NLRC5"
if g in GB.index and g in expr.index:
    x = pd.to_numeric(GB.loc[g, sh], errors="coerce")
    yv = pd.to_numeric(expr.loc[g, sh], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(yv)
    r, p = spearmanr(x[ok], yv[ok])
    ax.scatter(x[ok], yv[ok], s=4, alpha=0.35, c="#2166AC", edgecolors="none")
    b = np.polyfit(x[ok], yv[ok], 1)
    xs = np.linspace(x[ok].min(), x[ok].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="#B2182B", lw=1.2)
    ax.set_xlabel(f"{g} promoter β"); ax.set_ylabel(f"{g} expression (log2)")
    ax.set_title(f"g  Exemplar: {g}\nρ={r:.2f}, p={p:.1e}", loc="left", fontweight="bold")

for a in fig.axes:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
plt.savefig(f"{F}/Fig6_chromatin.pdf", dpi=600, bbox_inches="tight")
plt.savefig(f"{F}/Fig6_chromatin.png", dpi=600, bbox_inches="tight")
plt.close()
print("Fig6 ok")
