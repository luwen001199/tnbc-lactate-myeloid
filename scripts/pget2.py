#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pget2.py -- resumable, verified multi-threaded chunked downloader.
Each chunk is written to <out>.part<i> and its byte length is verified;
the loop repeats until all chunks are complete, then concatenates.
Usage: python3 pget2.py <url> <outfile> [n_threads]
"""
import sys, os, threading, time, urllib.request, json

CH = 4 * 1024 * 1024   # 4 MB chunks


def get_size(url):
    """Prefer HEAD; fall back to a 2-byte ranged GET and read Content-Range."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    for _ in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range", "")
                if "/" in cr:
                    return int(cr.split("/")[-1])
                if r.headers.get("Content-Length"):
                    return int(r.headers["Content-Length"])
        except Exception:
            time.sleep(3)
    raise RuntimeError("size probe failed")


def expected(i, total):
    s = i * CH
    e = min(total - 1, s + CH - 1)
    return s, e, e - s + 1


def download_chunk(url, i, total, outdir):
    s, e, n = expected(i, total)
    p = os.path.join(outdir, f"part{i}")
    if os.path.exists(p) and os.path.getsize(p) == n:
        return True
    hdr = {"Range": f"bytes={s}-{e}"}
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=300) as r, open(p, "wb") as f:
            while True:
                b = r.read(262144)
                if not b:
                    break
                f.write(b)
    except Exception:
        pass
    return os.path.exists(p) and os.path.getsize(p) == n


def main():
    url, out = sys.argv[1], sys.argv[2]
    nthreads = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    total = get_size(url)
    nchunks = (total + CH - 1) // CH
    outdir = out + ".parts"
    os.makedirs(outdir, exist_ok=True)
    print(f"size={total/1e6:.1f}MB chunks={nchunks} threads={nthreads}", flush=True)
    t0 = time.time()
    for rnd in range(40):
        todo = [i for i in range(nchunks)
                if not (os.path.exists(os.path.join(outdir, f"part{i}"))
                        and os.path.getsize(os.path.join(outdir, f"part{i}")) == expected(i, total)[2])]
        if not todo:
            break
        print(f"round {rnd}: {len(todo)} chunks to fetch", flush=True)
        lock = threading.Lock()
        idx = [0]
        tot = [len(todo)]

        def work():
            while True:
                with lock:
                    if idx[0] >= len(todo):
                        return
                    i = todo[idx[0]]
                    idx[0] += 1
                ok = download_chunk(url, i, total, outdir)
                if not ok:
                    with lock:
                        todo.append(i)
                with lock:
                    if idx[0] % 20 == 0:
                        done = len([1 for j in range(nchunks)
                                    if os.path.exists(os.path.join(outdir, f"part{j}"))
                                    and os.path.getsize(os.path.join(outdir, f"part{j}")) == expected(j, total)[2]])
                        print(f"   {done}/{nchunks} chunks  {done/nchunks*100:.1f}%  "
                              f"{done*CH/1e6/(time.time()-t0+1):.2f} MB/s", flush=True)
        ths = [threading.Thread(target=work) for _ in range(nthreads)]
        [t.start() for t in ths]
        [t.join() for t in ths]
    # concatenate
    with open(out, "wb") as fo:
        for i in range(nchunks):
            p = os.path.join(outdir, f"part{i}")
            if not (os.path.exists(p) and os.path.getsize(p) == expected(i, total)[2]):
                raise RuntimeError(f"chunk {i} incomplete after retries")
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
    print(f"done {os.path.getsize(out)/1e6:.1f} MB in {time.time()-t0:.0f}s -> {out}", flush=True)
    # optional md5 check
    if len(sys.argv) > 4:
        import hashlib
        h = hashlib.md5()
        with open(out, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        print("md5", h.hexdigest(), "expected", sys.argv[4],
              "MATCH" if h.hexdigest() == sys.argv[4] else "MISMATCH", flush=True)


if __name__ == "__main__":
    main()
