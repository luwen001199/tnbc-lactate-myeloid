#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_genepanels.py
================
TNBC myeloid "lactate-trained immunity" project -- GENE PANELS (curated, literature-anchored)

Three literature-curated pools + two control modules.
Each panel records: gene symbol, module, and the biological rationale,
so every gene is traceable to a citable source (see PANEL_SOURCES).

Output: /workspace/results/gene_panels.tsv  (long format: module, gene)
        /workspace/results/gene_panels.json (module -> list of genes)
"""
import json, os

OUT = "/workspace/results"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# POOL 1. Lactate metabolism / glycolytic lactate production & transport
#         (the "lactate microenvironment" axis)
# Rationale: TNBC is the most glycolytic breast subtype; lactate is exported via
#            MCT1/MCT4 and imported by myeloid cells, fuelling histone lactylation
#            through the lactyl-CoA / p300 axis.
# ----------------------------------------------------------------------------
LACTATE_METABOLISM = [
    # glycolysis (upstream lactate supply)
    "SLC2A1", "SLC2A3", "HK1", "HK2", "HK3", "GPI", "PFKL", "PFKM", "PFKP",
    "PFKFB3", "PFKFB4", "ALDOA", "ALDOC", "TPI1", "GAPDH", "PGK1", "PGAM1",
    "ENO1", "ENO2", "PKM", "PKLR",
    # lactate production / interconversion
    "LDHA", "LDHB", "LDHC", "LDHAL6A",
    # lactate transport
    "SLC16A1", "SLC16A3", "SLC16A4", "SLC16A7", "SLC16A8", "BSG", "CD147",
    # lactate -> lactyl-CoA & TCA shunting
    "ACLY", "ACSS1", "ACSS2", "ACSS3", "SUCLG1", "SUCLG2", "PDHA1", "PDHB",
    "PDK1", "PDK2", "PDK3", "PDK4", "PDP1", "MPC1", "MPC2",
    # hypoxia / master glycolytic drivers
    "HIF1A", "EPAS1", "MYC", "MTOR", "SLC1A5", "GLS", "GLUD1",
]
LACTATE_METABOLISM = [g for g in LACTATE_METABOLISM if g != "CD147"]

# tight "lactate core" -- used for the single-cell lactyl-score proxy
LACTATE_CORE = ["LDHA", "LDHB", "SLC16A1", "SLC16A3", "SLC2A1", "HK2",
                "PFKFB3", "PKM", "ENO1", "PGK1", "ALDOA", "PDK1", "ACLY",
                "GAPDH", "TPI1", "PGAM1", "BSG", "HIF1A"]

# ----------------------------------------------------------------------------
# POOL 2. Lactylation machinery: writers (lactyltransferases) / erasers
#         (delactylases) / readers (YEATS-domain)
# Rationale: p300/CBP are the established histone lactyltransferases;
#            HDAC1-3 and SIRT1-3 act as delactylases; AF9/ENL (YEATS) read H3K9la;
#            HDACi (vorinostat / tucidinostat) pharmacologically target the erasers.
# ----------------------------------------------------------------------------
KLA_WRITERS = ["EP300", "CREBBP", "KAT2A", "KAT2B", "KAT5", "KAT6A", "KAT7",
               "KAT8", "HAT1", "NCOA1", "NCOA3"]
KLA_ERASERS = ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6", "HDAC7",
               "HDAC8", "HDAC9", "HDAC10", "HDAC11", "SIRT1", "SIRT2", "SIRT3",
               "SIRT6", "SIRT7"]
KLA_READERS = ["MLLT1", "MLLT3", "YEATS2", "YEATS4", "BRD2", "BRD3", "BRD4",
               "BRD9", "ATAD2", "SP140", "ZMYND8", "DPF2"]
KLA_MACHINERY = sorted(set(KLA_WRITERS + KLA_ERASERS + KLA_READERS))

# ----------------------------------------------------------------------------
# POOL 3A. ANTI-TUMOUR "trained-like" myeloid programme
#          (IRF8-dependent, HDACi-retrainable, MHC-II-high, CCL8-CCR5-CD8 axis)
# ----------------------------------------------------------------------------
TRAINED_ANTI = [
    # ---- IRF8 / BATF3-driven trained myeloid identity
    "IRF8", "BATF3", "ZEB2", "ID2", "SPI1", "CEBPA", "CEBPB", "CIITA",
    "NLRC5", "STAT1", "STAT2", "IRF1", "IRF5", "IRF7", "RELB",
    # ---- MHC-II / antigen presentation (trained hallmark: enhanced secondary response)
    "HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1",
    "HLA-DQB1", "HLA-DMA", "HLA-DMB", "HLA-DOA", "HLA-DOB", "CD74",
    "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9", "PSME1", "PSME2", "CTSS",
    "LGMN", "LAMP3", "HLA-A", "HLA-B", "HLA-C", "B2M", "NLRC5",
    # ---- MHC-I machinery / cross-presentation
    "CD80", "CD86", "CD40", "ICAM1", "TNFSF9", "SLAMF7",
    # ---- trained-immunity cytokines & CD8 chemoattraction (CCL8-CCR5, CXCL9/10-CXCR3)
    "CCL8", "CCR5", "CCL5", "CCL3", "CCL4", "CXCL9", "CXCL10", "CXCL11",
    "CXCR3", "IL12A", "IL12B", "IL15", "IL18", "TNF", "IL1B", "IL1A",
    "IL6", "IL23A", "IFNG", "TNFSF10",
    # ---- trained pattern-recognition / effector machinery
    "TLR2", "TLR4", "TLR7", "TLR8", "NOD2", "CARD9", "NLRP3", "CLEC7A",
    "MYD88", "TICAM1", "TBK1", "MAVS", "DDX58", "IFIH1", "CGAS", "STING1",
    # ---- Fc receptors / phagocytosis / oxidative burst
    "FCGR1A", "FCGR1B", "FCGR2A", "FCGR2B", "FCGR3A", "FCGR3B", "FCER1G",
    "ITGAM", "ITGAX", "ITGB2", "CD36", "MARCO", "MSR1", "SCARB1", "MERTK",
    "CYBB", "NCF1", "NCF2", "NCF4", "MPO", "LYZ", "CTSB", "CTSD",
    # ---- cDC1 / monocytic-dendritic bridge (IRF8-dependent)
    "CLEC9A", "XCR1", "CADM1", "WDFY4", "SNX22",
]
# NOTE: the metabolic rewiring genes (MYC/HIF1A/SLC2A1/AKT1/MTOR/RPS6KB1/RPTOR...)
# are deliberately kept OUT of TRAINED_ANTI and placed in TRAINED_METABOLIC below,
# so that the "anti-tumour trained" axis is collinear neither with the lactate axis
# nor with myeloid abundance.
TRAINED_METABOLIC = ["MYC", "HIF1A", "SLC2A1", "AKT1", "MTOR", "RPS6KB1",
                     "EIF4EBP1", "RPTOR", "MLST8", "PFKFB3", "HK2", "LDHA",
                     "SLC16A1", "PDK1", "ACLY"]
TRAINED_ANTI = sorted(set(g for g in TRAINED_ANTI))

# ----------------------------------------------------------------------------
# POOL 3B. PRO-TUMOUR "lactate-trained memory" myeloid programme
#          (M2-like / TREM2+ lipid-TAM / MDSC-like / TGFbeta-IL10 immunosuppression)
# ----------------------------------------------------------------------------
TRAINED_PRO = [
    # ---- M2-like / alternatively activated
    "MRC1", "CD163", "CLEC10A", "CD209", "SIGLEC1", "VSIG4", "IL4I1",
    "FOLR2", "LYVE1", "GPNMB", "TREM2", "SPP1", "FN1", "VCAN", "MMP9",
    "MMP12", "MMP14", "TIMP1", "SERPINE1", "THBS1", "PLAU",
    # ---- immunosuppressive cytokines / checkpoints
    "TGFB1", "TGFB2", "TGFB3", "IL10", "IL4", "IL13", "IDO1", "IDO2",
    "CD274", "PDCD1LG2", "LGALS1", "LGALS3", "LGALS9", "HAVCR2", "VSIR",
    "ENTPD1", "NT5E", "ARG1", "ARG2", "PTGS2", "IL6R", "IL4R",
    # ---- tumour-promoting chemokines / angiogenesis / ECM
    "CCL2", "CCL7", "CCL13", "CCL17", "CCL18", "CCL20", "CCL22", "CXCL8",
    "CXCL12", "CXCR4", "VEGFA", "VEGFB", "PDGFB", "WNT5A", "TGFBI",
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "FN1", "SPARC",
    # ---- MDSC-like / immature monocytic
    "S100A8", "S100A9", "S100A12", "FCN1", "VCAN", "CD14", "THBS1", "OLFM4",
    "RETN", "SELL", "CSF3R", "IL1R2", "CLEC4E", "CLEC5A", "ANXA1", "ANXA2",
    "SAT1", "G0S2", "PYCARD", "NAMPT", "WARS1", "PLBD1", "MGST1", "TXN",
    # ---- STAT3 / NF-kB tolerised programme
    "STAT3", "STAT6", "NFKB1", "NFKB2", "RELA", "SOCS1", "SOCS3", "PIM1",
    "BCL3", "TNFAIP3", "NFKBIA", "MYD88", "IRAK3", "TNIP1", "TRAF1",
    # ---- efferocytosis / lipid handling (TREM2-TAM)
    "TREM2", "TYROBP", "APOE", "APOC1", "LPL", "CD36", "FABP4", "FABP5",
    "PLIN2", "ABCA1", "ABCG1", "MSR1", "MRC1",
]
TRAINED_PRO = sorted(set(TRAINED_PRO))

# ----------------------------------------------------------------------------
# POOL 4. Innate / antiviral (interferon-driven) control module
#         Distinguishes "trained" from "antiviral/ISG" innate memory
# ----------------------------------------------------------------------------
INNATE_ANTIVIRAL = ["OAS1", "OAS2", "OAS3", "OASL", "GBP1", "GBP2", "GBP3",
                    "GBP4", "GBP5", "ZBP1", "IFIT1", "IFIT2", "IFIT3", "IFIT5",
                    "ISG15", "ISG20", "MX1", "MX2", "IFI44", "IFI44L", "IFI6",
                    "IFI27", "IFI35", "IRF7", "STAT1", "STAT2", "DDX58",
                    "IFIH1", "USP18", "RSAD2", "CMPK2", "HERC5", "SAMD9L",
                    "EPSTI1", "PARP9", "DTX3L", "TRIM22", "LY6E", "BST2"]

# ----------------------------------------------------------------------------
# POOL 5. HDACi-induced (IRF8-dependent) retraining signature
#         Used for the "pharmacological reprogramming" analysis
# ----------------------------------------------------------------------------
HDACI_RETRAIN = ["IRF8", "BATF3", "ZEB2", "ID2", "CIITA", "HLA-DRA",
                 "HLA-DRB1", "CD74", "CD80", "CD86", "FCGR1A", "TAP1",
                 "PSMB8", "PSMB9", "CCL8", "CCR5", "CXCL9", "CXCL10",
                 "IL12B", "IL15", "STAT1", "NLRC5", "CLEC9A", "XCR1",
                 "TNFSF10", "IL1B", "TNF", "SPI1", "CEBPB", "MAVS", "TBK1"]

# ----------------------------------------------------------------------------
# Marker sets for single-cell annotation of the myeloid compartment
# ----------------------------------------------------------------------------
SC_MARKERS = {
    "Monocyte_classical":  ["LYZ", "CD14", "S100A8", "S100A9", "VCAN", "FCN1",
                            "LST1", "AIF1", "TYROBP", "FCGR3A", "CSF3R", "SELL"],
    "Monocyte_nonclassical": ["FCGR3A", "MS4A7", "CDKN1C", "RHOC", "HES4",
                              "MEF2C", "LST1", "CST3", "TCF7L2", "MTSS1"],
    "TAM":                  ["C1QA", "C1QB", "C1QC", "CD68", "CD163", "MRC1",
                             "TREM2", "GPNMB", "APOE", "FOLR2", "LYVE1",
                             "SPP1", "FN1", "CSF1R", "MRC1", "SIGLEC1"],
    "TAM_M1like":           ["IL1B", "TNF", "CXCL9", "CXCL10", "CCL5", "IL6",
                             "CD80", "CD86", "HLA-DRA", "IRF1", "STAT1",
                             "NLRP3", "SOD2", "CXCL11", "IDO1"],
    "TAM_M2like":           ["MRC1", "CD163", "CLEC10A", "VSIG4", "FOLR2",
                             "LYVE1", "TREM2", "GPNMB", "IL10", "TGFB1",
                             "CCL18", "FN1", "SPP1", "MMP9", "ARG1"],
    "cDC1":                 ["CLEC9A", "XCR1", "CADM1", "BATF3", "IRF8",
                             "WDFY4", "C1ORF54", "SNX22"],
    "cDC2":                 ["CD1C", "FCER1A", "CLEC10A", "SIRPA", "ITGAX",
                             "HLA-DQA1", "CLEC4A"],
    "pDC":                  ["LILRA4", "CLEC4C", "IL3RA", "GZMB", "SERPINF1",
                             "TCF4", "IRF7", "SPIB"],
    "MDSC_like":            ["S100A8", "S100A9", "S100A12", "FCN1", "VCAN",
                             "CD14", "IL1R2", "CLEC4E", "OLFM4", "RETN",
                             "THBS1", "SAT1", "CSF3R"],
    "Proliferating_myeloid": ["MKI67", "TOP2A", "PCNA", "TYMS", "STMN1",
                              "CDK1", "BIRC5", "UBE2C"],
    "T_CD8":                ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "NKG7",
                             "CCL5", "IFNG", "GZMK", "KLRD1"],
    "NK":                   ["NKG7", "GNLY", "NKG2D", "KLRD1", "KLRF1", "GZMB",
                             "PRF1", "FCGR3A", "NCR1"],
    "B_cell":               ["MS4A1", "CD79A", "CD79B", "IGHM", "BANK1", "IGKC"],
    "T_CD4":                ["IL7R", "CCR7", "LTB", "TRAC", "CD3D", "FOXP3"],
    "Epithelial":           ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT5"],
    "Fibroblast":           ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB",
                             "ACTA2", "FAP"],
    "Endothelial":          ["PECAM1", "VWF", "CLDN5", "PLVAP", "CDH5"],
}

PANELS = {
    "LACTATE_METABOLISM": LACTATE_METABOLISM,
    "LACTATE_CORE": LACTATE_CORE,
    "KLA_WRITERS": KLA_WRITERS,
    "KLA_ERASERS": KLA_ERASERS,
    "KLA_READERS": KLA_READERS,
    "KLA_MACHINERY": KLA_MACHINERY,
    "TRAINED_ANTI": TRAINED_ANTI,
    "TRAINED_METABOLIC": TRAINED_METABOLIC,
    "TRAINED_PRO": TRAINED_PRO,
    "INNATE_ANTIVIRAL": INNATE_ANTIVIRAL,
    "HDACI_RETRAIN": HDACI_RETRAIN,
    **{f"SC_{k}": v for k, v in SC_MARKERS.items()},
}

PANEL_SOURCES = {
    "LACTATE_METABOLISM": "KEGG hsa00010 Glycolysis/Gluconeogenesis; HALLMARK_GLYCOLYSIS; lactate shuttle (MCT1/MCT4) literature",
    "LACTATE_CORE": "LDHA/MCT/p300 histone-lactylation axis in TNBC (2024-2025 multi-omics)",
    "KLA_WRITERS": "p300/CBP histone lactyltransferase (Zhang et al. Nature 2019); GCN5/KAT2A lactyltransferase",
    "KLA_ERASERS": "HDAC1-3 & SIRT1-3 delactylase activity; HDACi (vorinostat/tucidinostat) targets",
    "KLA_READERS": "YEATS-domain H3K9la readers AF9(MLLT1)/ENL(MLLT3)/YEATS2/GAS41; BET bromodomain",
    "TRAINED_ANTI": "IRF8-dependent trained immunity (Nat Immunol); BCG/beta-glucan trained-monocyte signatures; HDACi-retrained TAM. Metabolic genes excluded to avoid collinearity with the lactate axis",
    "TRAINED_METABOLIC": "Glycolytic/mTOR rewiring that mechanistically *underpins* trained immunity (kept as a separate module)",
    "TRAINED_PRO": "M2/TREM2+ lipid-TAM and MDSC-associated immunosuppression; tolerised STAT3/NF-kB programme",
    "INNATE_ANTIVIRAL": "Interferon-stimulated innate module (OAS2/GBP4/ZBP1) used as non-trained innate control",
    "HDACI_RETRAIN": "HDAC-inhibitor-induced IRF8-dependent antitumour trained macrophage programme",
}

if __name__ == "__main__":
    rows = []
    for m, genes in PANELS.items():
        for g in genes:
            rows.append((m, g))
    with open(f"{OUT}/gene_panels.tsv", "w") as f:
        f.write("module\tgene\n")
        for m, g in rows:
            f.write(f"{m}\t{g}\n")
    with open(f"{OUT}/gene_panels.json", "w") as f:
        json.dump({"panels": {m: sorted(set(g)) for m, g in PANELS.items()},
                   "sources": PANEL_SOURCES}, f, indent=2)
    print(f"{'module':32s} n_genes")
    for m, g in PANELS.items():
        print(f"{m:32s} {len(set(g))}")
    print("\n[written]", f"{OUT}/gene_panels.tsv", f"{OUT}/gene_panels.json")
