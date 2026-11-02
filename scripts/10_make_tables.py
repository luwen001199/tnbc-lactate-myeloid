#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10_make_tables.py -- build all manuscript tables from the result files."""
import json
import numpy as np, pandas as pd

R, M = "/workspace/results", "/workspace/manuscript"
cox = pd.read_csv(f"{R}/cox_multivariable.csv")
sc = pd.read_csv(f"{R}/sc_myeloid_tnbc_comparison.csv")
arch = pd.read_csv(f"{R}/epi_promoter_architecture_panels.csv")
enr = pd.read_csv(f"{R}/epi_architecture_enrichment.csv")
progm = pd.read_csv(f"{R}/epi_programme_methylation_tnbc.csv")
lat = pd.read_csv(f"{R}/epi_latched_by_programme.csv", index_col=0)
panels = json.load(open(f"{R}/gene_panels.json"))["panels"]
sub = pd.read_csv(f"{R}/subtype_assignments.csv", index_col=0)
tcga = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
for c in [x for x in sub.columns if x.startswith("subtype_")]:
    tcga[c] = sub[c].reindex(tcga.index)


def fp(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def est(r):
    return f"{r.hr_sd:.2f} ({r.lo:.2f}–{r.hi:.2f})"


out = []
out.append("# Tables\n")

# ---------------------------------------------------------------- Table 1
out.append("\n## Table 1 — Multivariable Cox models, METABRIC (adjusted for age, nodal positivity, NPI and chemotherapy)\n")
out.append("| Stratum | Endpoint | n / events | Variable | HR per 1 SD (95% CI) | p |")
out.append("|---|---|---|---|---|---|")
d = cox[(cox.cohort == "METABRIC") & (cox.model == "adjusted")].copy()
d["grp"] = d.scope.map({"TNBC": "TNBC", "ALL": "All subtypes"})
for _, r in d.sort_values(["grp", "endpoint", "p"]).iterrows():
    out.append(f"| {r.grp} | {r.endpoint.upper()} | {int(r.n)} / {int(r.events)} | "
               f"`{r['var']}` | {est(r)} | {fp(r.p)} |")
out.append("\nModel A in Table 2 notation; covariates were not used to construct any score.")

# ---------------------------------------------------------------- Table 2
out.append("\n## Table 2 — Single-cell comparison of myeloid programmes, TNBC vs non-TNBC (GSE176078)\n")
out.append("| Cell type | Programme | n (TNBC / other) | Δ (TNBC − other) | p |")
out.append("|---|---|---|---|---|")
name = {"lactyl": "Lactate core", "trained_pro": "Pro-tumour trained",
        "trained_anti": "Anti-tumour trained", "apc_score": "Antigen presentation",
        "m2_score": "M2 / suppressive", "hdaci_retrain": "HDACi retraining",
        "kla_writer": "Lactylation writers", "kla_reader": "Lactylation readers",
        "antiviral": "Innate antiviral (control)"}
for ct in ["Monocyte", "Macrophage", "DCs", "Cycling_Myeloid"]:
    s = sc[(sc.celltype == ct) & (sc.score.isin(name))]
    for _, r in s.sort_values("p").iterrows():
        out.append(f"| {ct} | {name[r.score]} | {int(r.n_TNBC)} / {int(r.n_other)} | "
                   f"{r.delta:+.3f} | {fp(r.p)} |")

# ---------------------------------------------------------------- Table 3
out.append("\n## Table 3 — Promoter chromatin architecture of the five gene panels\n")
out.append("| Programme | n genes | DHS open (%) | CpG island (%) | Monocyte H3K27ac, donor A / B (%) | Monocyte H3K4me3, donor A / B (%) |")
out.append("|---|---|---|---|---|---|")
order = ["Anti-tumour trained", "Pro-tumour suppressive", "Lactate metabolism",
         "Lactylation machinery", "Innate antiviral (control)"]
A = arch.set_index("programme")
for p in order:
    if p not in A.index:
        continue
    r = A.loc[p]
    out.append(f"| {p} | {int(r.n_genes)} | {r.open_promoter_pct:.1f} | {r.cpg_island_pct:.1f} | "
               f"{r.monoA_k27ac_pct:.1f} / {r.monoB_k27ac_pct:.1f} | "
               f"{r.monoA_k4me3_pct:.1f} / {r.monoB_k4me3_pct:.1f} |")
out.append(f"\nBackground (all 28,237 RefSeq promoters): DHS open 94.9%, CpG island 59.4%, "
           f"monocyte H3K27ac 44.9% / 51.5%, monocyte H3K4me3 54.0% / 62.0%.")

# ---------------------------------------------------------------- Table 4
out.append("\n## Table 4 — Programme-level promoter methylation in TNBC (LT-pro high vs LT-anti high)\n")
out.append("| Programme | n genes | β (LT-pro high) | β (LT-anti high) | Δβ | p | ρ with LT-pro | p(ρ) |")
out.append("|---|---|---|---|---|---|---|---|")
for _, r in progm.iterrows():
    out.append(f"| {r.programme} | {int(r.n_genes)} | {r.beta_pro:.3f} | {r.beta_anti:.3f} | "
               f"{r.delta:+.4f} | {fp(r.p)} | {r.rho_LTpro:+.3f} | {fp(r.p_rho)} |")

# ---------------------------------------------------------------- Table 5
out.append("\n## Table 5 — 'Latched' promoter configuration by programme\n")
out.append("| Programme | Genes in panel | Latched | % |")
out.append("|---|---|---|---|")
for p in order:
    if p not in lat.index:
        continue
    r = lat.loc[p]
    out.append(f"| {p} | {int(r.n)} | {int(r.latched)} | {100*r.latched/r.n:.1f} |")
out.append("\nLatched = CpG-island promoter + DNase/monocyte-H3K27ac overlap + mean β < 0.3 "
           "+ promoter-methylation/expression ρ < −0.1.")

# ---------------------------------------------------------------- Table S1
out.append("\n## Table S1 — Validation of the mRNA-based TNBC definition (TCGA-BRCA)\n")
t = tcga.dropna(subset=["pam50"])
tn = t[t.is_tnbc == True]
basal = (tn.pam50.str.lower() == "basal").sum()
n_lab = tn.pam50.notna().sum()
out.append("| Metric | Value |")
out.append("|---|---|")
out.append(f"| Tumours with expression + clinical data | {len(tcga)} |")
out.append(f"| mRNA-defined TNBC (GMM cutoffs on ESR1/PGR/ERBB2) | {int(tcga.is_tnbc.sum())} "
           f"({100*tcga.is_tnbc.mean():.1f}%) |")
out.append(f"| mRNA-defined TNBC with a PAM50 label | {int(n_lab)} |")
out.append(f"| Of these, PAM50 basal | {int(basal)} ({100*basal/max(n_lab,1):.1f}%) |")
out.append("\nGMM cutoffs: ESR1 10.156, PGR 8.685, ERBB2 15.124 (log₂ RSEM). "
           "The Pan-Cancer Atlas PAM50 label for the basal-like class is recorded as \"Basal\".")

# ---------------------------------------------------------------- Table S2
out.append("\n## Table S2 — Gene panels\n")
for k in ["LACTATE_METABOLISM", "LACTATE_CORE", "KLA_WRITERS", "KLA_ERASERS",
          "KLA_READERS", "KLA_MACHINERY", "TRAINED_ANTI", "TRAINED_PRO",
          "TRAINED_METABOLIC", "INNATE_ANTIVIRAL", "HDACI_RETRAIN"]:
    if k in panels:
        g = panels[k]
        out.append(f"\n**{k}** (n = {len(g)}): {', '.join(g)}")

# ---------------------------------------------------------------- Table S3
out.append("\n## Table S3 — Complete Cox output (both cohorts, univariable and adjusted)\n")
out.append("| Cohort | Stratum | Endpoint | Model | n / events | Variable | HR per 1 SD (95% CI) | p |")
out.append("|---|---|---|---|---|---|---|---|")
d = cox.copy().sort_values(["cohort", "scope", "endpoint", "model", "p"])
for _, r in d.iterrows():
    out.append(f"| {r.cohort} | {r.scope} | {r.endpoint.upper()} | {r.model} | "
               f"{int(r.n)} / {int(r.events)} | `{r['var']}` | {est(r)} | {fp(r.p)} |")

# ------------------------------------------------- Table S4: PH assumption
ph = pd.read_csv(f"{R}/cox_sensitivity_ph.csv")
out.append("\n## Table S4 — Proportional-hazards assumption (rank-transformed Schoenfeld test)\n")
out.append("| Cohort | Stratum | Endpoint | Variable | n / events | p |")
out.append("|---|---|---|---|---|---|")
for _, r in ph.sort_values(["cohort", "endpoint", "scope", "variable"]).iterrows():
    out.append(f"| {r.cohort} | {r.scope} | {r.endpoint.upper()} | `{r.variable}` | "
               f"{int(r.n)} / {int(r.events)} | {fp(r.ph_p)} |")
out.append("\nLT-pro and the lactate components violate the assumption; LT-anti, "
           "trained-anti and the HDACi-retraining module do not.")

# ------------------------------------- Table S5: landmark + RMST sensitivity
sens = pd.read_csv(f"{R}/cox_sensitivity_landmark_rmst.csv")
out.append("\n## Table S5 — Landmark analysis (split at 5 years) and 10-year restricted mean "
           "survival time (RMST), METABRIC overall survival\n")
out.append("| Stratum | Variable | Analysis | HR per 1 SD (95% CI) | p | n / events | RMST (months) |")
out.append("|---|---|---|---|---|---|---|")
for _, r in sens.sort_values(["scope", "variable", "analysis", "window"]).iterrows():
    hr = "—" if pd.isna(r.hr) else f"{r.hr:.2f} ({r.lo:.2f}–{r.hi:.2f})"
    p = "—" if pd.isna(r.p) else fp(r.p)
    rmst = f"{r.rmst:.1f}" if "rmst" in sens.columns and pd.notna(r.get("rmst")) else "—"
    out.append(f"| {r.scope} | `{r.variable}` | {r.analysis} | {hr} | {p} | "
               f"{int(r.n)} / {int(r.events)} | {rmst} |")
out.append("\nLandmark windows: events after 5 years censored at 5 years (early window); "
           "patients still at risk at 5 years re-baselined (late window). RMST computed by "
           "tertile of the axis at a 10-year horizon.")

# ------------------------------------- Table S6: MCP-counter sensitivity
ovl = pd.read_csv(f"{R}/mcpcounter_marker_overlap.csv")
out.append("\n## Table S6 — Sensitivity of the infiltrate adjustment to the abundance estimator\n")
out.append("**(a) Overlap between the curated marker panels and the official MCP-counter "
           "signatures (Becht et al. 2016)**\n")
out.append("| Population | Curated markers (n) | Official MCP-counter markers (n) | Shared | "
           "Jaccard | Official markers available in METABRIC |")
out.append("|---|---|---|---|---|---|")
for _, r in ovl.iterrows():
    out.append(f"| {r.population} | {int(r.n_curated)} | {int(r.n_official)} | {int(r.shared)} | "
               f"{r.jaccard:.2f} | {int(r.metabric_official_covered)} |")

cox_s = pd.read_csv(f"{R}/mcpcounter_sensitivity_cox.csv")
out.append("\n**(b) Multivariable Cox models, METABRIC, adjusted for age, nodal positivity, "
           "NPI and chemotherapy — curated versus official abundance**\n")
out.append("| Stratum | Endpoint | Variable | Curated HR per 1 SD (95% CI) | p | "
           "Official HR per 1 SD (95% CI) | p | n / events |")
out.append("|---|---|---|---|---|---|---|---|")
for v in ["LT_pro", "LT_anti", "trained_anti_adj"]:
    sub = cox_s[cox_s.variable == v]
    for (sc, ep), g in sub.groupby(["scope", "endpoint"], sort=False):
        c = g[g.abundance == "curated"].iloc[0]
        o = g[g.abundance == "official"].iloc[0]
        out.append(f"| {sc} | {ep.upper()} | `{v}` | {c.hr:.2f} ({c.lo:.2f}–{c.hi:.2f}) | {fp(c.p)} | "
                   f"{o.hr:.2f} ({o.lo:.2f}–{o.hi:.2f}) | {fp(o.p)} | {int(c.n)} / {int(c.events)} |")
out.append("\nBoth estimators are linear in the marker genes (mean expression, optionally "
           "z-scaled), and residualisation is invariant to the affine step, so the only "
           "substantive difference between the columns is the marker gene set. Myeloid "
           "abundance from the two sets correlated at ρ = 0.878 in TCGA (n = 1,097); the "
           "resulting LT-pro axes at ρ = 0.952 and LT-anti at ρ = 0.715 in METABRIC; the "
           "quadrant rule reproduced the subtype assignment in 70.6% of METABRIC tumours.")

open(f"{M}/tables.md", "w").write("\n".join(out) + "\n")

# supplementary file = S1..S5 only
sup = ["# Supplementary Tables\n"]
for line in out:
    if line.startswith("## Table S"):
        sup.append(line)
    elif sup and sup[-1].count("\n") >= 0:
        pass
# simpler: split out from the first "## Table S" heading
full = "\n".join(out)
i = full.index("## Table S1")
open(f"{M}/tables_supplementary.md", "w").write("# Supplementary Tables\n\n" + full[i:] + "\n")
print("wrote", f"{M}/tables.md", "and", f"{M}/tables_supplementary.md")
