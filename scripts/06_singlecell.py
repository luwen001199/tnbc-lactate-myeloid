#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_singlecell.py
================
Single-cell analysis of GSE176078 (Wu et al. 2021, Nat Genet; 100,064 cells,
26 treatment-naive breast cancers: TNBC n=42,512 cells; ER+ 38,241; HER2+ 19,311).

Uses the authors' own high-confidence cell-type and tumour-subtype annotations,
so no re-annotation noise is introduced.

Analyses
--------
1  myeloid compartment: lactyl / pro-trained / anti-trained scores per cell
2  TNBC vs non-TNBC comparison within monocytes, macrophages, DCs
3  identification of the pro-tumour lactate-high / antigen-presentation-low TAM state
4  monocyte -> macrophage pseudotime with LDHA/EP300/IRF8/CCL8/HLA-DRA dynamics
5  CellChat-style ligand-receptor inference from myeloid subsets to CD8 T cells
6  per-sample myeloid state fractions vs bulk LT_pro axis
"""
import os, json
import numpy as np, pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

D, R, F = "/workspace/data", "/workspace/results", "/workspace/figures"
os.makedirs(F, exist_ok=True)
sc.settings.verbosity = 1
sc.settings.n_jobs = 8
P = json.load(open(f"{R}/gene_panels.json"))["panels"]

# ------------------------------------------------------------------ load
M = sp.load_npz(f"{D}/sc176078_counts.npz")           # genes x cells
genes = [l.strip() for l in open(f"{D}/sc176078/Wu_etal_2021_BRCA_scRNASeq/count_matrix_genes.tsv")]
bc = [l.strip() for l in open(f"{D}/sc176078/Wu_etal_2021_BRCA_scRNASeq/count_matrix_barcodes.tsv")]
meta = pd.read_csv(f"{D}/sc176078/Wu_etal_2021_BRCA_scRNASeq/metadata.csv", index_col=0)
# NOTE: metadata index is already "CID####_BARCODE", identical to count_matrix_barcodes.tsv
print(f"[load] matrix {M.shape}, genes {len(genes)}, barcodes {len(bc)}, meta {meta.shape}")

keep = [i for i, b in enumerate(bc) if b in set(meta.index)]
print(f"[load] matched cells: {len(keep)}")
A = sc.AnnData(M[:, keep].T.tocsr())
A.obs_names = [bc[i] for i in keep]
A.var_names = genes
A.var_names_make_unique()
A.obs = meta.reindex(A.obs_names)
print(f"[load] anndata {A.shape}")
print(A.obs.celltype_major.value_counts())

# ------------------------------------------------------------------ QC + normalise + scores
CACHE = f"{R}/sc_all_scored.h5ad"
if os.path.exists(CACHE):
    A = sc.read_h5ad(CACHE)
    print(f"[cache] reloaded scored anndata {A.shape}")
else:
    A.var["mt"] = A.var_names.str.startswith("MT-")
    # the authors' metadata already carries nFeature_RNA / percent.mito
    if "percent.mito" in A.obs.columns:
        A.obs["percent_mito"] = pd.to_numeric(A.obs["percent.mito"], errors="coerce")
    else:
        sc.pp.calculate_qc_metrics(A, qc_vars=["mt"], percent_top=None,
                                   log1p=False, inplace=True)
        A.obs["percent_mito"] = A.obs["pct_counts_mt"]
    A.obs["nFeature_RNA"] = pd.to_numeric(A.obs["nFeature_RNA"], errors="coerce")
    A = A[(A.obs.nFeature_RNA > 200) & (A.obs.percent_mito < 20)].copy()
    A.layers["counts"] = A.X.copy()
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)
    A.raw = A
    print(f"[qc] {A.shape}")

    def add_score(ad, geneset, name, n_ctrl=50, seed=0):
        gs = [g for g in geneset if g in ad.var_names]
        if len(gs) < 5:
            print(f"   [warn] {name}: only {len(gs)} genes")
            return
        sc.tl.score_genes(ad, gs, score_name=name, ctrl_size=n_ctrl, random_state=seed)

    A.obs["is_TNBC"] = A.obs.subtype.eq("TNBC")
    for nm, gs in [("lactyl", P["LACTATE_CORE"]), ("lactate_full", P["LACTATE_METABOLISM"]),
                   ("kla_writer", P["KLA_WRITERS"]), ("kla_eraser", P["KLA_ERASERS"]),
                   ("kla_reader", P["KLA_READERS"]),
                   ("trained_anti", P["TRAINED_ANTI"]), ("trained_pro", P["TRAINED_PRO"]),
                   ("hdaci_retrain", P["HDACI_RETRAIN"]), ("antiviral", P["INNATE_ANTIVIRAL"])]:
        add_score(A, gs, nm)

    # antigen presentation / M2 sub-modules
    APC = ["HLA-DRA", "HLA-DRB1", "HLA-DMA", "HLA-DMB", "CD74", "TAP1", "TAP2",
           "PSMB8", "PSMB9", "CIITA", "CD80", "CD86"]
    M2 = ["MRC1", "CD163", "VSIG4", "FOLR2", "LYVE1", "TREM2", "GPNMB", "IL10",
          "TGFB1", "CCL18", "FN1", "SPP1"]
    add_score(A, APC, "apc_score"); add_score(A, M2, "m2_score")

    A.write(CACHE)
    print("[save] sc_all_scored.h5ad")

# ensure downstream columns exist whether loaded from cache or computed
if "is_TNBC" not in A.obs.columns:
    A.obs["is_TNBC"] = A.obs.subtype.eq("TNBC")

# ------------------------------------------------------------------ myeloid compartment
MY = A.obs.celltype_major.eq("Myeloid")
myel = A[MY].copy()
print(f"\n[myeloid] {myel.shape[0]} cells")
print(myel.obs.celltype_minor.value_counts())

out_rows = []
for minor in ["Monocyte", "Macrophage", "DCs", "Cycling_Myeloid"]:
    sub = myel[myel.obs.celltype_minor.eq(minor)]
    if sub.shape[0] < 30:
        continue
    a = sub[sub.obs.is_TNBC]
    b = sub[~sub.obs.is_TNBC]
    for s in ["lactyl", "trained_anti", "trained_pro", "hdaci_retrain", "apc_score",
              "m2_score", "kla_writer", "kla_reader", "antiviral"]:
        if s not in sub.obs:
            continue
        u, p = mannwhitneyu(a.obs[s], b.obs[s])
        out_rows.append(dict(celltype=minor, score=s, n_TNBC=len(a), n_other=len(b),
                             mean_TNBC=a.obs[s].mean(), mean_other=b.obs[s].mean(),
                             delta=a.obs[s].mean() - b.obs[s].mean(), p=p))
cmp_df = pd.DataFrame(out_rows)
cmp_df["fdr"] = np.minimum(1, cmp_df.p * len(cmp_df))   # conservative BH-free bound
cmp_df.to_csv(f"{R}/sc_myeloid_tnbc_comparison.csv", index=False)
print("\n[myeloid] TNBC vs non-TNBC:")
print(cmp_df.sort_values("p").head(24).to_string(index=False))

# per-gene check of the core mechanistic genes
core = ["LDHA", "SLC16A1", "SLC16A3", "EP300", "CREBBP", "HDAC1", "HDAC2", "IRF8",
        "CCL8", "CCR5", "HLA-DRA", "CD74", "TREM2", "MRC1", "CD163", "SPP1",
        "TGFB1", "CXCL9", "CXCL10"]
rows = []
for g in core:
    if g not in myel.var_names:
        continue
    v = np.asarray(myel[:, g].X.todense()).ravel()
    df = pd.DataFrame(dict(v=v, tnbc=myel.obs.is_TNBC.values, ct=myel.obs.celltype_minor.values))
    for ct in ["Monocyte", "Macrophage", "DCs"]:
        d = df[df.ct == ct]
        a, b = d.v[d.tnbc], d.v[~d.tnbc]
        if len(a) > 20 and len(b) > 20:
            u, p = mannwhitneyu(a, b)
            rows.append(dict(gene=g, celltype=ct, mean_TNBC=a.mean(),
                             mean_other=b.mean(), log2FC=np.log2((a.mean() + 1e-4) / (b.mean() + 1e-4)), p=p))
gd = pd.DataFrame(rows)
gd.to_csv(f"{R}/sc_myeloid_gene_level.csv", index=False)
print("\n[genes] core mechanistic genes (myeloid):")
print(gd.sort_values("p").head(20).to_string(index=False))

# ------------------------------------------------------------------ TAM state definition
mac = myel[myel.obs.celltype_minor.eq("Macrophage")].copy()
print(f"\n[tam] macrophages: {mac.shape[0]}")
mac.obs["lac_hi"] = mac.obs["lactyl"] > mac.obs["lactyl"].median()
mac.obs["apc_lo"] = mac.obs["apc_score"] < mac.obs["apc_score"].median()
mac.obs["state"] = "other"
mac.obs.loc[mac.obs.lac_hi & mac.obs.apc_lo, "state"] = "lactate-high/APC-low (pro-tumour)"
mac.obs.loc[(~mac.obs.lac_hi) & (~mac.obs.apc_lo), "state"] = "lactate-low/APC-high (anti-tumour)"
frac = pd.crosstab(mac.obs.subtype, mac.obs.state, normalize="index") * 100
print("\n[tam] state fraction by tumour subtype (%):")
print(frac.round(1).to_string())
frac.to_csv(f"{R}/sc_tam_state_fraction.csv")

# ------------------------------------------------------------------ pseudotime mono -> macrophage
print("\n[pseudotime] monocyte -> macrophage trajectory")
traj = myel[myel.obs.celltype_minor.isin(["Monocyte", "Macrophage"])].copy()
GE = ["LDHA", "SLC16A1", "EP300", "IRF8", "CCL8", "HLA-DRA", "CD74", "TREM2",
      "MRC1", "SPP1", "TGFB1", "CXCL9"]
# gene values taken from the FULL myeloid object BEFORE HVG filtering
gex = pd.DataFrame({g: np.asarray(myel[:, g].X.todense()).ravel()
                    for g in GE if g in myel.var_names}, index=myel.obs_names)

sc.pp.highly_variable_genes(traj, n_top_genes=1500)
traj = traj[:, traj.var.highly_variable].copy()
sc.pp.scale(traj, max_value=10)
sc.tl.pca(traj, n_comps=30, svd_solver="arpack")
sc.pp.neighbors(traj, n_neighbors=15)
sc.tl.diffmap(traj, n_comps=15)
# deterministic root: monocyte most extreme along the diffusion component that
# best separates Monocyte from Macrophage
dm = traj.obsm["X_diffmap"]
is_mono = (traj.obs.celltype_minor == "Monocyte").values
sep = np.abs(dm[is_mono].mean(0) - dm[~is_mono].mean(0))
k = int(np.argmax(sep))
mono_vals = dm[is_mono, k]
sgn = 1.0 if dm[is_mono, k].mean() < dm[~is_mono, k].mean() else -1.0
root_idx = int(np.where(is_mono)[0][np.argmax(sgn * mono_vals)])
traj.uns["iroot"] = root_idx
print(f"   root = diffusion comp {k}, cell {traj.obs_names[root_idx]} "
      f"({traj.obs.celltype_minor.iloc[root_idx]})")
sc.tl.dpt(traj)
traj.obs["pseudotime"] = traj.obs["dpt_pseudotime"]
# bin along pseudotime, average key genes
pt = traj.obs[["pseudotime", "celltype_minor", "subtype", "is_TNBC"]].copy()
for g in gex.columns:
    pt[g] = gex.loc[traj.obs_names, g].values
pt.to_csv(f"{R}/sc_pseudotime_gene_dynamics.csv")
print(pt.groupby(pd.qcut(pt.pseudotime, 10, duplicates="drop"))[
    [c for c in ["LDHA", "EP300", "IRF8", "CCL8", "HLA-DRA", "TREM2"] if c in pt.columns]
].mean().round(3).to_string())

# correlation of pseudotime with scores (stratified by subtype)
for s in ["lactyl", "trained_anti", "trained_pro"]:
    for tn in [True, False]:
        d = pt[pt.is_TNBC == tn].dropna(subset=["pseudotime"])
        if s in myel.obs.columns:
            v = myel[traj.obs_names].obs[s].reindex(d.index)
            r, p = spearmanr(d.pseudotime, v)
            print(f"   {s:14s} {'TNBC' if tn else 'non-TNBC'}: rho={r:+.3f} p={p:.2e}")

# ------------------------------------------------------------------ ligand-receptor inference
print("\n[lr] CellChat-style inference: myeloid -> CD8 T")
LR = [("CCL8", "CCR5"), ("CCL5", "CCR5"), ("CCL3", "CCR5"), ("CCL4", "CCR5"),
      ("CXCL9", "CXCR3"), ("CXCL10", "CXCR3"), ("CXCL11", "CXCR3"),
      ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"), ("TGFB2", "TGFBR1"),
      ("IL6", "IL6R"), ("TNF", "TNFRSF1A"), ("IL1B", "IL1R1"),
      ("LGALS9", "HAVCR2"), ("CD274", "PDCD1"), ("PDCD1LG2", "PDCD1"),
      ("SPP1", "CD44"), ("MIF", "CD74"), ("SLC16A1", "SLC16A1"),
      ("VEGFA", "FLT1"), ("IL10", "IL10RB"), ("CCL2", "CCR2"),
      ("TNFSF10", "TNFRSF10B"), ("IL15", "IL2RB"), ("CXCL12", "CXCR4")]
lr = [(l, r) for l, r in LR if l in A.var_names and r in A.var_names]
print(f"   {len(lr)} LR pairs evaluable")

cd8 = A[A.obs.celltype_major.eq("T-cells")].copy()
# CD8 identification by marker score
sc.tl.score_genes(cd8, ["CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "CCL5"], score_name="cd8s")
cd8 = cd8[cd8.obs.cd8s > cd8.obs.cd8s.median()].copy()
print(f"   CD8-like T cells: {cd8.shape[0]}")

rows = []
for tn in [True, False]:
    for minor in ["Monocyte", "Macrophage", "DCs"]:
        send = myel[(myel.obs.celltype_minor == minor) & (myel.obs.is_TNBC == tn)]
        recv = cd8[cd8.obs.is_TNBC == tn]
        if send.shape[0] < 50 or recv.shape[0] < 50:
            continue
        for l, rr in lr:
            lv = np.asarray(send[:, l].X.todense()).ravel()
            rv = np.asarray(recv[:, rr].X.todense()).ravel()
            # CellChat-like: product of mean expression (mass-action)
            score = lv.mean() * rv.mean()
            rows.append(dict(subtype="TNBC" if tn else "non-TNBC", sender=minor,
                             ligand=l, receptor=rr, lig_mean=lv.mean(),
                             rec_mean=rv.mean(), lr_score=score))
lrd = pd.DataFrame(rows)
lrd.to_csv(f"{R}/sc_lr_myeloid_cd8.csv", index=False)
piv = lrd.pivot_table(index=["sender", "ligand", "receptor"], columns="subtype",
                      values="lr_score").fillna(0)
piv["log2FC_TNBC_vs_other"] = np.log2((piv.get("TNBC", 0) + 1e-3) / (piv.get("non-TNBC", 0) + 1e-3))
print("\n[lr] top differential myeloid->CD8 axes (TNBC vs non-TNBC):")
print(piv.sort_values("log2FC_TNBC_vs_other", ascending=False).head(15).round(3).to_string())

print("\n[done] 06_singlecell.py")
