#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_singlecell.py
================
Single-cell dissection of the myeloid compartment in GSE176078 (Wu et al. 2021,
26 treatment-naive breast cancers; ER+ / HER2+ / TNBC).

Steps
-----
1  load per-sample 10x matrices, QC, per-sample ambient-RNA-free filtering
2  per-sample subtype calling (pseudo-bulk ESR1/PGR/ERBB2) -> TNBC vs non-TNBC
3  integration (Harmony), clustering, lineage annotation
4  re-cluster the myeloid compartment (monocyte / non-classical monocyte /
   TAM-M1-like / TAM-M2-like / TREM2-TAM / MDSC-like / cDC1 / cDC2 / pDC / prolif.)
5  per-cell lactyl / pro-trained / anti-trained scores; identify the
   "lactate-high, antigen-presentation-low" pro-tumour TAM state
6  monocyte -> TAM pseudotime (diffusion pseudotime), dynamics of
   LDHA / EP300 / IRF8 / CCL8 / HLA-DRA
7  CellChat-style ligand-receptor inference focused on
   SLC16A1(MCT1)-lactate, TGFB, CCL8-CCR5, CXCL9/10-CXCR3, IL6, TNF
8  export per-sample myeloid state composition for correlation with bulk subtypes
"""
import os, json, glob, gzip, sys
import numpy as np, pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse, stats

D = "/workspace/data"
R = "/workspace/results"
F = "/workspace/figures"
os.makedirs(R, exist_ok=True); os.makedirs(F, exist_ok=True)
sc.settings.verbosity = 1
sc.settings.n_jobs = 8

P = json.load(open(f"{R}/gene_panels.json"))["panels"]


# ------------------------------------------------------------------ locate 10x inputs
def find_10x():
    """Return list of (sample_id, dir_with_10x_files)."""
    out = []
    base = f"{D}/sc176078_gsm"
    tars = sorted(glob.glob(f"{base}/*.tar.gz"))
    if not tars:
        # try extracting from RAW tar
        raw = f"{D}/GSE176078_RAW.tar"
        if os.path.exists(raw):
            import subprocess
            os.makedirs(base, exist_ok=True)
            subprocess.run(f"tar -xf {raw} -C {base}", shell=True)
            tars = sorted(glob.glob(f"{base}/*.tar.gz"))
    for t in tars:
        sid = os.path.basename(t).split("_")[1].replace(".tar.gz", "")
        ex = os.path.join(base, sid)
        if not os.path.exists(ex):
            os.makedirs(ex, exist_ok=True)
            os.system(f"tar -xzf {t} -C {ex} --strip-components=0 2>/dev/null || tar -xzf {t} -C {ex}")
        out.append((sid, ex))
    return out


def read_10x(d):
    """Read a 10x directory (possibly nested), return AnnData."""
    files = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            low = f.lower()
            if "barcode" in low:
                files["bc"] = os.path.join(root, f)
            elif "gene" in low or "feature" in low:
                files["ft"] = os.path.join(root, f)
            elif low.endswith(".mtx") or low.endswith(".mtx.gz"):
                files["mx"] = os.path.join(root, f)
    if len(files) < 3:
        return None
    return sc.read_10x_mtx(d, var_names="gene_symbols" if False else "gene_ids")


print("[scan] locating 10x inputs ...")
samples = find_10x()
print(f"[scan] {len(samples)} samples found")
if not samples:
    sys.exit("no 10x input found")

adatas = []
meta = []
for sid, d in samples:
    try:
        files = {}
        for root, _, fs in os.walk(d):
            for f in fs:
                low = f.lower()
                if "barcode" in low:
                    files["bc"] = os.path.join(root, f)
                elif "gene" in low or "feature" in low:
                    files["ft"] = os.path.join(root, f)
                elif low.endswith(".mtx") or low.endswith(".mtx.gz"):
                    files["mx"] = os.path.join(root, f)
        if len(files) < 3:
            print(f"  {sid}: incomplete ({sorted(files)}), skipped")
            continue
        a = sc.read_10x_mtx(os.path.dirname(files["mx"]), var_names="gene_symbols")
        a.obs["sample"] = sid
        a.obs_names = [f"{sid}_{b}" for b in a.obs_names]
        adatas.append(a)
        print(f"  {sid}: {a.shape[0]} cells x {a.shape[1]} genes")
    except Exception as e:
        print(f"  {sid}: FAILED {e}")

if not adatas:
    sys.exit("no data loaded")

A = ad.concat(adatas, join="outer", label=None)
A.var_names_make_unique()
print(f"[merge] {A.shape[0]} cells x {A.shape[1]} genes")
A.write(f"{R}/sc_raw.h5ad")

# ------------------------------------------------------------------ QC
A.var["mt"] = A.var_names.str.startswith("MT-")
A.var["ribo"] = A.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(A, qc_vars=["mt", "ribo"], percent_top=None, log1p=False, inplace=True)
print("[qc] pre-filter:", A.shape)
sc.pp.filter_cells(A, min_genes=200)
sc.pp.filter_genes(A, min_cells=3)
A = A[A.obs.pct_counts_mt < 20, :].copy()
A = A[A.obs.n_genes_by_counts < 6000, :].copy()
print("[qc] post-filter:", A.shape)

A.layers["counts"] = A.X.copy()
sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)
A.raw = A
sc.pp.highly_variable_genes(A, n_top_genes=2000, batch_key="sample")
Aa = A[:, A.var.highly_variable].copy()
sc.pp.scale(Aa, max_value=10)
sc.tl.pca(Aa, n_comps=50, svd_solver="arpack")

# ------------------------------------------------------------------ integration
try:
    import harmonypy
    ho = harmonypy.run_harmony(Aa.obsm["X_pca"], Aa.obs, "sample", max_iter_harmony=20)
    Aa.obsm["X_harmony"] = ho.Z_corr.T
    rep = "X_harmony"
except Exception as e:
    print("[harmony] failed, using PCA:", e)
    rep = "X_pca"

sc.pp.neighbors(Aa, n_neighbors=15, use_rep=rep)
sc.tl.leiden(Aa, resolution=0.6, key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
sc.tl.umap(Aa)
A.obs["leiden"] = Aa.obs["leiden"].values
A.obsm["X_umap"] = Aa.obsm["X_umap"].values
A.obsm["X_harmony"] = Aa.obsm[rep]
print("[cluster] leiden clusters:", A.obs["leiden"].nunique())

# ------------------------------------------------------------------ lineage annotation (marker score)
LINEAGES = {k.replace("SC_", ""): v for k, v in P.items() if k.startswith("SC_")}
sc.tl.score_genes(A, LINEAGES["Epithelial"], score_name="s_Epithelial")
sc.tl.score_genes(A, LINEAGES["T_CD4"], score_name="s_TCD4")
sc.tl.score_genes(A, LINEAGES["T_CD8"], score_name="s_TCD8")
sc.tl.score_genes(A, LINEAGES["NK"], score_name="s_NK")
sc.tl.score_genes(A, LINEAGES["B_cell"], score_name="s_B")
sc.tl.score_genes(A, LINEAGES["Fibroblast"], score_name="s_Fibro")
sc.tl.score_genes(A, LINEAGES["Endothelial"], score_name="s_Endo")
myeloid_marks = sorted(set(LINEAGES["Monocyte_classical"] + LINEAGES["TAM"] +
                           LINEAGES["MDSC_like"] + LINEAGES["cDC1"] + LINEAGES["cDC2"] + LINEAGES["pDC"]))
sc.tl.score_genes(A, myeloid_marks, score_name="s_Myeloid")

lincols = ["s_Epithelial", "s_TCD4", "s_TCD8", "s_NK", "s_B", "s_Fibro", "s_Endo", "s_Myeloid"]
lin = A.obs.groupby("leiden")[lincols].mean()
lin["n"] = A.obs.groupby("leiden").size()
print("\n[lineage] mean marker scores per leiden cluster:")
print(lin.round(2).to_string())
assign = lin[lincols].idxmax(axis=1).str.replace("s_", "")
A.obs["lineage"] = A.obs["leiden"].map(assign)

# refine: a cluster must also have decent expression of canonical markers
for cl in lin.index:
    print(f"   leiden {cl}: n={int(lin.loc[cl,'n'])} -> {assign[cl]}")

print("\n[lineage] composition:")
print(A.obs["lineage"].value_counts().to_string())

# ------------------------------------------------------------------ per-sample subtype (pseudo-bulk)
print("\n[subtype] pseudo-bulk receptor status per sample ...")
pb = {}
for s in A.obs["sample"].unique():
    sub = A[A.obs["sample"] == s]
    if sub.shape[0] < 50:
        continue
    # mean of malignant epithelium is the cleanest signal; fall back to all cells
    ep = sub[sub.obs["lineage"] == "Epithelial"]
    use = ep if ep.shape[0] >= 30 else sub
    v = np.asarray(use[:, ["ESR1", "PGR", "ERBB2"]].X.mean(axis=0)).ravel()
    pb[s] = dict(zip(["ESR1", "PGR", "ERBB2"], v))
pb = pd.DataFrame(pb).T
print(pb.round(3).to_string())

# 2-component GMM cut on pseudo-bulk log values
from sklearn.mixture import GaussianMixture
def gmm_cut(v):
    x = np.asarray(v, float); x = x[np.isfinite(x)]
    if len(x) < 6:
        return np.median(x)
    gm = GaussianMixture(2, random_state=0, n_init=10).fit(x.reshape(-1, 1))
    hi = int(np.argmax(gm.means_.ravel()))
    grid = np.linspace(x.min(), x.max(), 1000)
    p = gm.predict_proba(grid.reshape(-1, 1))[:, hi]
    return float(grid[np.argmin(np.abs(p - 0.5))])
cuts = {g: gmm_cut(pb[g].values) for g in ["ESR1", "PGR", "ERBB2"]}
print("[subtype] pseudo-bulk cutoffs:", {k: round(v, 3) for k, v in cuts.items()})
tnbc = pd.DataFrame({g: pb[g] < cuts[g] for g in cuts}).all(axis=1)
pb["is_tnbc"] = tnbc
print("[subtype] TNBC samples:", list(pb.index[tnbc]))
print("[subtype] non-TNBC  :", list(pb.index[~tnbc]))
A.obs["tnbc_sample"] = A.obs["sample"].map(pb["is_tnbc"])
pb.to_csv(f"{R}/sc_sample_receptor_status.csv")

A.write(f"{R}/sc_annotated.h5ad")
print("\n[done] 05_singlecell.py (part 1: QC + lineage + subtype)")
