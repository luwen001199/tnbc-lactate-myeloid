#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download GSE176078 per-sample 10x tars, N files in parallel."""
import os, subprocess, sys, concurrent.futures, re

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176078/suppl/"
OUT = "/workspace/data/sc176078_gsm"
os.makedirs(OUT, exist_ok=True)
files = []
for line in open("/workspace/data/filelist_GSE176078.txt"):
    p = line.rstrip("\n").split("\t")
    if p and p[0] == "File" and p[1].endswith(".tar.gz"):
        files.append(p[1])

def one(f, nthreads=6):
    dst = os.path.join(OUT, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 500000:
        return f, "cached"
    r = subprocess.run(["/root/biovenv/bin/python", "/workspace/scripts/pget2.py",
                        BASE + f, dst, str(nthreads)], capture_output=True, text=True)
    return f, r.stdout.strip().splitlines()[-1] if r.stdout else "FAIL"

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    for f, st in ex.map(one, files):
        print(f"{f}: {st}", flush=True)
print("ALL DONE")
