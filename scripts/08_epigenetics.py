#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_epigenetics.py
=================
Chromatin-level test of the "epigenetic latch" hypothesis.

Rationale / data availability
-----------------------------
No open-access per-patient ATAC-seq exists for TCGA-BRCA (the Corces 2018
TCGA-ATAC BAMs on GDC are dbGaP-controlled, ~20 GB each), and ENCODE has no
human monocyte/macrophage ATAC release.  We therefore test the chromatin layer
with two orthogonal, fully public readouts:

  (i)  PER-PATIENT chromatin state -- Illumina HM450 promoter DNA methylation
       (TCGA-BRCA, UCSC Xena) measured in the SAME 1,097 tumours (212 TNBC) for
       which the transcriptome scores were derived.  Promoter methylation is a
       mitotically heritable mark and is anti-correlated with accessibility, so
       LT-pro-high vs LT-anti-high differences at IRF8 / antigen-presentation /
       lactate loci are a genuine per-patient chromatin comparison.

  (ii) STATIC chromatin architecture -- UCSC clustered DNase I hypersensitivity
       (125 cell types), UCSC CpG islands, and Roadmap Epigenomics consolidated
       H3K27ac / H3K4me3 peaks in primary CD14+ monocytes (E029, E124) -- i.e.
       is the locus an active, high-CpG, open promoter in the myeloid lineage?

Analyses
--------
A  promoter openness / CpG-island / monocyte H3K27ac+H3K4me3 by programme
B  Fisher enrichment vs all RefSeq promoters
C  per-patient promoter methylation: LT-pro-high vs LT-anti-high (and vs LT-pro
   low / LT-anti low), plus continuous Spearman with LT_pro and LT_anti
D  methylation-expression coupling (is the mark regulatory?)
E  "latched memory genes": CpG-island + open + hypomethylated + negatively
   coupled to expression
"""
import os, json, gzip, subprocess, tempfile
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact

D, R, F = "/workspace/data", "/workspace/results", "/workspace/figures"
os.makedirs(F, exist_ok=True)
P = json.load(open(f"{R}/gene_panels.json"))["panels"]

# programmes: curated core loci (mechanistic read-out) + full panels (statistics)
GROUPS = {
    "Anti-tumour trained": ["IRF8", "BATF3", "CIITA", "HLA-DRA", "HLA-DRB1", "CD74",
                            "TAP1", "TAP2", "PSMB8", "PSMB9", "CD80", "CD86", "CCL8",
                            "CCR5", "CXCL9", "CXCL10", "FCGR1A", "CLEC9A", "XCR1", "ZEB2"],
    "Lactate metabolism": ["LDHA", "LDHB", "SLC16A1", "SLC16A3", "SLC2A1", "HK2",
                           "PFKFB3", "ENO1", "PGK1", "ALDOA", "PDK1", "ACLY", "BSG", "HIF1A"],
    "Pro-tumour suppressive": ["TGFB1", "IL10", "MRC1", "CD163", "TREM2", "SPP1",
                               "ARG1", "IDO1", "CCL2", "VEGFA", "STAT3", "CXCL12", "FN1"],
    "Lactylation machinery": ["EP300", "CREBBP", "HDAC1", "HDAC2", "HDAC3", "MLLT1",
                              "MLLT3", "YEATS2", "SIRT1"],
    "Innate antiviral (control)": ["OAS1", "OAS2", "GBP4", "ZBP1", "IFIT1", "IFIT3",
                                   "ISG15", "MX1", "IRF7", "STAT1", "STAT2"],
}
PANEL_GROUPS = {
    "Anti-tumour trained": P["TRAINED_ANTI"],
    "Pro-tumour suppressive": P["TRAINED_PRO"],
    "Lactate metabolism": P["LACTATE_METABOLISM"],
    "Lactylation machinery": P["KLA_MACHINERY"],
    "Innate antiviral (control)": P["INNATE_ANTIVIRAL"],
}

# ================================================================== 1. HM450 manifest
print("[1] HM450 manifest (GPL13534) ...")
man = pd.read_csv(f"{D}/HM450_manifest.csv.gz", skiprows=7, low_memory=False,
                  usecols=["IlmnID", "CHR", "MAPINFO", "UCSC_RefGene_Name",
                           "UCSC_RefGene_Group", "Relation_to_UCSC_CpG_Island", "DHS"])
print("   ", man.shape)
man = man.dropna(subset=["UCSC_RefGene_Name"])
man["gene"] = man["UCSC_RefGene_Name"].astype(str).str.split(";")
m = man.explode("gene")[["IlmnID", "CHR", "MAPINFO", "gene",
                         "UCSC_RefGene_Group", "Relation_to_UCSC_CpG_Island", "DHS"]]
m = m[(m.gene != "") & (m.gene != "nan")]
print(f"   probe-gene pairs {len(m)} | probes {m.IlmnID.nunique()}")

PROM = {"TSS1500", "TSS200", "1stExon", "5'UTR"}
mp = m[m["UCSC_RefGene_Group"].isin(PROM)].copy()
print(f"   promoter probe-gene pairs {len(mp)}")

# ================================================================== 2. coordinates
print("[2] refGene TSS / DNase clusters / CpG islands / Roadmap monocyte peaks ...")
rg = pd.read_csv(f"{D}/refGene.hg19.txt.gz", sep="\t", header=None,
                 usecols=[2, 3, 4, 5, 12], names=["chrom", "strand", "txStart", "txEnd", "gene"])
rg = rg[rg.chrom.str.match(r"^chr[\dXYM]+$", na=False)]
rg["tss"] = np.where(rg.strand == "+", rg.txStart, rg.txEnd)
# canonical TSS = transcript whose TSS is closest to the median TSS of the gene
med = rg.groupby("gene").tss.transform("median")
rg["d"] = (rg.tss - med).abs()
rg = rg.sort_values(["gene", "d"]).drop_duplicates("gene")
tss = rg[["gene", "chrom", "tss"]].reset_index(drop=True)
print(f"   genes with TSS: {len(tss)}")


def load_bed(path, cols=(0, 1, 2), names=("chrom", "start", "end"), val_col=None):
    """generic bed reader -> dict chrom -> float ndarray [start, end, (score)]"""
    if not os.path.exists(path):
        return {}
    usecols = list(cols) + ([val_col] if val_col is not None else [])
    nm = list(names) + (["score"] if val_col is not None else [])
    df = pd.read_csv(path, sep="\t", header=None, usecols=usecols, names=nm,
                     dtype={nm[0]: str})
    df = df[df["chrom"].astype(str).str.match(r"^chr[\dXYM]+$", na=False)]
    out = {}
    for c, s in df.groupby("chrom"):
        out[c] = s[nm[1:]].values.astype(float)
    return out


# UCSC "clustered DNase I hypersensitivity" -- bin,chrom,start,end,n_celltypes,max_signal,...
dn = load_bed(f"{D}/dnaseClustered.hg19.txt.gz", cols=(1, 2, 3),
              names=("chrom", "start", "end"), val_col=4)
print(f"   DNase clusters: {sum(len(v) for v in dn.values())}")
# UCSC cpgIslandExt -- bin,chrom,start,end,...
cpg = load_bed(f"{D}/cpgIslandExt.hg19.txt.gz", cols=(1, 2, 3),
               names=("chrom", "start", "end"))
print(f"   CpG islands: {sum(len(v) for v in cpg.values())}")
MONO_MARKS = {}
for tag, fn in [("monoA_k27ac", "E029-H3K27ac.narrowPeak.gz"),   # E029 primary CD14+ monocytes
                ("monoA_k4me3", "E029-H3K4me3.narrowPeak.gz"),
                ("monoB_k27ac", "E124-H3K27ac.narrowPeak.gz"),   # E124 CD14+ monocytes, donor 2
                ("monoB_k4me3", "E124-H3K4me3.narrowPeak.gz")]:
    p = f"{D}/{fn}"
    if os.path.exists(p):
        # ENCODE narrowPeak: chrom(0) start(1) end(2) name(3) score(4)
        MONO_MARKS[tag] = load_bed(p, cols=(0, 1, 2),
                                   names=("chrom", "start", "end"), val_col=4)
        print(f"   {tag} peaks: {sum(len(v) for v in MONO_MARKS[tag].values())}")

# ================================================================== 3. promoter annotation
WIN = 2000
print(f"[3] annotating promoters (TSS ± {WIN} bp) ...")


def annotate(tss_df, intervals, value=False):
    """returns (n_overlapping, sum_of_scores) per row"""
    n = np.zeros(len(tss_df), dtype=float)
    s = np.zeros(len(tss_df), dtype=float)
    lo_all = (tss_df.tss.values - WIN)
    hi_all = (tss_df.tss.values + WIN)
    pos = {c: g.index.values for c, g in tss_df.groupby("chrom")}
    for chrom, idx in pos.items():
        itv = intervals.get(chrom)
        if itv is None or len(itv) == 0:
            continue
        st, en = itv[:, 0], itv[:, 1]
        vl = itv[:, 2] if (value and itv.shape[1] > 2) else np.ones(len(st))
        o = np.argsort(st)
        st, en, vl = st[o], en[o], vl[o]
        me = np.maximum.accumulate(en)
        for i in idx:
            lo, hi = lo_all[i], hi_all[i]
            j = int(np.searchsorted(st, hi, side="right")) - 1
            cnt = 0; tot = 0.0
            while j >= 0 and me[j] >= lo:
                if en[j] >= lo:
                    cnt += 1; tot += vl[j]
                j -= 1
            n[i] = cnt; s[i] = tot
    return n, s


tss["lo"] = tss.tss - WIN
tss["hi"] = tss.tss + WIN
n_, s_ = annotate(tss, dn, value=True)
tss["dhs_n"] = n_; tss["dhs_score"] = s_
n_, _ = annotate(tss, cpg)
tss["cpg_island"] = n_ > 0
for tag, iv in MONO_MARKS.items():
    n_, s_ = annotate(tss, iv, value=True)
    tss[tag + "_n"] = n_
    tss[tag + "_score"] = s_
prom = tss.drop(columns=["lo", "hi"])
prom.to_csv(f"{R}/promoter_annotation.csv", index=False)
print(f"   annotated {len(prom)} promoters")

# ================================================================== 4. architecture by programme
def arch_tab(groups):
    rows = []
    for gname, genes in groups.items():
        sub = prom[prom.gene.isin(set(genes))]
        if sub.empty:
            continue
        r = dict(programme=gname, n_genes=len(sub),
                 open_promoter_pct=100 * (sub.dhs_n > 0).mean(),
                 median_dhs_score=sub.dhs_score.median(),
                 cpg_island_pct=100 * sub.cpg_island.mean())
        for tag in MONO_MARKS:
            r[f"{tag}_pct"] = 100 * (sub[tag + "_n"] > 0).mean()
        rows.append(r)
    return pd.DataFrame(rows)


arch = arch_tab(GROUPS)
print("\n[4] promoter chromatin architecture, curated core loci:")
print(arch.round(1).to_string(index=False))
arch.to_csv(f"{R}/epi_promoter_architecture_core.csv", index=False)

archp = arch_tab(PANEL_GROUPS)
print("\n[4b] promoter chromatin architecture, FULL panels:")
print(archp.round(1).to_string(index=False))
archp.to_csv(f"{R}/epi_promoter_architecture_panels.csv", index=False)

bg_open = float((prom.dhs_n > 0).mean())
bg_cpg = float(prom.cpg_island.mean())
print(f"\n   background (all {len(prom)} RefSeq promoters): open {100*bg_open:.1f}% "
      f"| CpG-island {100*bg_cpg:.1f}%")
for tag in MONO_MARKS:
    print(f"   background {tag}: {100*(prom[tag+'_n']>0).mean():.1f}%")

enr = []
for groups, label in [(GROUPS, "core"), (PANEL_GROUPS, "panel")]:
    for gname, genes in groups.items():
        sub = prom[prom.gene.isin(set(genes))]
        if sub.empty:
            continue
        for feat, mask in [("open_DHS", (prom.dhs_n > 0).values),
                           ("CpG_island", prom.cpg_island.values)] + \
                          [(t, (prom[t + "_n"] > 0).values) for t in MONO_MARKS]:
            if feat == "open_DHS":
                sub_mask = (sub.dhs_n > 0).values
            elif feat == "CpG_island":
                sub_mask = sub.cpg_island.values
            else:
                sub_mask = (sub[feat + "_n"] > 0).values
            a = int(sub_mask.sum()); b = len(sub) - a
            c = int(mask.sum()); d = len(prom) - c
            orr, p = fisher_exact([[a, b], [c, d]])
            enr.append(dict(set=label, programme=gname, feature=feat,
                            n=len(sub), hit=a, pct=100 * a / len(sub),
                            bg_pct=100 * c / len(prom), OR=orr, p=p))
enr = pd.DataFrame(enr)
enr["fdr"] = np.minimum(1, enr.p * len(enr))
enr.to_csv(f"{R}/epi_architecture_enrichment.csv", index=False)
print("\n   Fisher enrichment vs all promoters (significant, FDR<0.05):")
print(enr[enr.fdr < 0.05].sort_values("p").round(4).to_string(index=False))

# ================================================================== 5. methylation extraction
print("\n[5] extracting TCGA-BRCA promoter methylation ...")
TARGET = sorted(set(sum(GROUPS.values(), [])) | set(sum(PANEL_GROUPS.values(), [])))
mp_t = mp[mp.gene.isin(set(TARGET))]
probes_needed = sorted(set(mp_t.IlmnID))
print(f"   target genes {len(TARGET)} | promoter probes {len(probes_needed)}")

METH = f"{D}/TCGA.BRCA.Meth450.gz"
OUT = f"{R}/meth_subset.tsv"
if not os.path.exists(OUT):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(probes_needed) + "\n")
        pfile = fh.name
    hdr = subprocess.run(f"zcat {METH} | head -1", shell=True, capture_output=True, text=True).stdout
    with open(OUT, "w") as fh:
        fh.write(hdr)
    cmd = f"zcat {METH} | grep -F -f {pfile} >> {OUT}"
    subprocess.run(cmd, shell=True, check=True)
    os.unlink(pfile)
    print("   subset written")
B = pd.read_csv(OUT, sep="\t", index_col=0)
print(f"   beta matrix {B.shape[0]} probes x {B.shape[1]} samples")
B.to_pickle(f"{R}/meth_probe_beta.pkl")

# gene-level promoter methylation = mean beta across its promoter probes
gene_beta = {}
for g, sub in mp_t.groupby("gene"):
    pr = [p for p in sub.IlmnID.unique() if p in B.index]
    if pr:
        gene_beta[g] = B.loc[pr].mean(axis=0)
GB = pd.DataFrame(gene_beta).T
GB.index = GB.index.astype(str)
GB.to_csv(f"{R}/meth_gene_promoter.csv")
print(f"   gene-level promoter methylation {GB.shape}")

# ================================================================== 6. per-patient comparison
print("\n[6] per-patient promoter methylation vs myeloid state ...")
sco = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
asg = pd.read_csv(f"{R}/subtype_assignments.csv", index_col=0)
for c in [x for x in asg.columns if x.startswith("subtype_")]:
    sco[c] = asg[c].reindex(sco.index)

common = [s for s in GB.columns if s in sco.index]
GBc = GB[common]
S = sco.loc[common]
print(f"   shared samples: {len(common)} (TNBC {int(S.is_tnbc.sum())})")


def quartile_sets(s, v):
    hi = s.index[v > v.quantile(0.75)]
    lo = s.index[v < v.quantile(0.25)]
    return hi, lo


def compare(stratum_mask, label):
    s = S[stratum_mask]
    gb = GBc[s.index]
    hi_pro, lo_pro = quartile_sets(s, s.LT_pro)
    hi_anti, lo_anti = quartile_sets(s, s.LT_anti)
    print(f"\n   [{label}] n={len(s)}  LT-pro hi/lo {len(hi_pro)}/{len(lo_pro)}  "
          f"LT-anti hi/lo {len(hi_anti)}/{len(lo_anti)}")
    rows = []
    for g in gb.index:
        v = gb.loc[g]
        rec = dict(gene=g)
        if len(hi_pro) > 15 and len(lo_pro) > 15:
            a, b = v[hi_pro].dropna(), v[lo_pro].dropna()
            _, p = mannwhitneyu(a, b)
            rec.update(delta_pro_hi_lo=a.mean() - b.mean(), p_pro=p)
        if len(hi_anti) > 15 and len(lo_anti) > 15:
            a, b = v[hi_anti].dropna(), v[lo_anti].dropna()
            _, p = mannwhitneyu(a, b)
            rec.update(delta_anti_hi_lo=a.mean() - b.mean(), p_anti=p)
        if len(hi_pro) > 15 and len(hi_anti) > 15:
            a, b = v[hi_pro].dropna(), v[hi_anti].dropna()
            _, p = mannwhitneyu(a, b)
            rec.update(delta_pro_vs_anti=a.mean() - b.mean(), p_pva=p)
        ok = np.isfinite(v)
        if ok.sum() > 50:
            r1, p1 = spearmanr(v[ok], s.LT_pro[ok])
            r2, p2 = spearmanr(v[ok], s.LT_anti[ok])
            rec.update(rho_LTpro=r1, p_LTpro=p1, rho_LTanti=r2, p_LTanti=p2)
        rec["programme"] = next((k for k, vl in GROUPS.items() if g in vl),
                                next((k for k, vl in PANEL_GROUPS.items() if g in vl), None))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(f"{R}/epi_methyl_by_state_{label}.csv", index=False)
    show = df.dropna(subset=["delta_pro_vs_anti"]).sort_values("p_pva")
    print(show.head(20)[["gene", "programme", "delta_pro_vs_anti", "p_pva",
                         "rho_LTpro", "rho_LTanti"]].round(4).to_string(index=False))
    return df


df_all = compare(pd.Series(True, index=S.index), "ALL")
df_tn = compare(S.is_tnbc.astype(bool), "TNBC")

# programme-level summary (TNBC): is the anti-tumour programme hypermethylated in
# the pro-tumour-lactate tumours?
print("\n[6b] programme-level promoter methylation (TNBC, LT-pro^hi vs LT-anti^hi):")
s = S[S.is_tnbc.astype(bool)]
gb = GBc[s.index]
hi_pro = s.index[s.LT_pro > s.LT_pro.quantile(0.75)]
hi_anti = s.index[s.LT_anti > s.LT_anti.quantile(0.75)]
rows = []
for gname, genes in PANEL_GROUPS.items():
    gg = [g for g in set(genes) if g in gb.index]
    if len(gg) < 5:
        continue
    v = gb.loc[gg].mean(axis=0)
    a, b = v[hi_pro].dropna(), v[hi_anti].dropna()
    _, p = mannwhitneyu(a, b)
    r, pr = spearmanr(v, s.LT_pro)
    rows.append(dict(programme=gname, n_genes=len(gg),
                     beta_pro=a.mean(), beta_anti=b.mean(),
                     delta=a.mean() - b.mean(), p=p,
                     rho_LTpro=r, p_rho=pr))
pl = pd.DataFrame(rows)
print(pl.round(4).to_string(index=False))
pl.to_csv(f"{R}/epi_programme_methylation_tnbc.csv", index=False)

# ================================================================== 7. methylation-expression coupling
print("\n[7] methylation-expression coupling ...")
expr = pd.read_pickle(f"{R}/expr_tcga_brca.pkl")
sh = [x for x in GBc.columns if x in expr.columns]
rows = []
for g in GBc.index:
    if g not in expr.index:
        continue
    v_m = pd.to_numeric(GBc.loc[g, sh], errors="coerce")
    v_e = pd.to_numeric(expr.loc[g, sh], errors="coerce")
    ok = np.isfinite(v_m) & np.isfinite(v_e)
    if ok.sum() < 50:
        continue
    r, p = spearmanr(v_m[ok], v_e[ok])
    rows.append(dict(gene=g, rho=r, p=p, n=int(ok.sum()),
                     programme=next((k for k, v in GROUPS.items() if g in v),
                                    next((k for k, v in PANEL_GROUPS.items() if g in v), None))))
me = pd.DataFrame(rows)
me.to_csv(f"{R}/epi_methylation_expression.csv", index=False)
print(me.sort_values("rho").head(15).round(4).to_string(index=False))
print("\n   median rho by programme:")
print(me.groupby("programme").rho.agg(["median", "mean", "count"]).round(3).to_string())

# ================================================================== 8. latched memory genes
print("\n[8] latched memory genes ...")
lat = prom.set_index("gene").join(me.set_index("gene")[["rho", "p"]], how="inner")
lat["meth_mean"] = GBc.mean(axis=1).reindex(lat.index)
gene2prog = {}
for k, v in GROUPS.items():
    for g in v:
        gene2prog.setdefault(g, k)
for k, v in PANEL_GROUPS.items():
    for g in v:
        gene2prog.setdefault(g, k)
lat["programme"] = [gene2prog.get(g) for g in lat.index]
lat = lat[lat.programme.notna() & (lat.programme != "Innate antiviral (control)")]
island = lat.cpg_island
mono_open = pd.Series(False, index=lat.index)
for t in MONO_MARKS:
    if t + "_n" in lat.columns:
        mono_open |= (lat[t + "_n"] > 0)
openp = (lat.dhs_n > 0) | mono_open
cand = lat[island & openp & (lat.meth_mean < 0.3) & (lat.rho < -0.1)].copy()
print(f"   CpG-island + open + hypomethylated (beta<0.3) + rho<-0.1: {len(cand)} genes")
cols = ["programme", "dhs_n", "dhs_score", "meth_mean", "rho", "p"]
cols = [c for c in cols if c in cand.columns]
print(cand.sort_values("rho")[cols].head(25).round(4).to_string())
cand.to_csv(f"{R}/epi_latched_memory_genes.csv")

# per-programme count of latched genes vs control
def _latched_count(d):
    return pd.Series(dict(
        n=len(d),
        latched=int((d.cpg_island & (d.dhs_n > 0) &
                     (d.meth_mean < 0.3) & (d.rho < -0.1)).sum())))


cnt = lat.groupby("programme", group_keys=False).apply(_latched_count)
print("\n   latched genes per programme:")
print(cnt.assign(pct=lambda d: 100 * d.latched / d.n).round(1).to_string())
cnt.to_csv(f"{R}/epi_latched_by_programme.csv")

print("\n[done] 08_epigenetics.py")
