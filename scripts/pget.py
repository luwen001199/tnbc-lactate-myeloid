#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pget.py -- multi-threaded HTTP range downloader (no external deps).
Usage: python3 pget.py <url> <outfile> [n_threads]
"""
import sys, os, threading, time, urllib.request

def get_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def worker(url, start, end, f, lock, idx, done):
    hdr = {"Range": f"bytes={start}-{end}"}
    req = urllib.request.Request(url, headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            buf = bytearray()
            while True:
                chunk = r.read(262144)
                if not chunk:
                    break
                buf += chunk
            with lock:
                f.seek(start)
                f.write(buf)
                done[0] += len(buf)
    except Exception as e:
        print(f"[thread {idx}] retry: {e}", flush=True)
        time.sleep(2)
        worker(url, start, end, f, lock, idx, done)

def main():
    url, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    total = get_size(url)
    print(f"size={total/1e6:.1f} MB  threads={n}", flush=True)
    with open(out, "wb") as f:
        f.truncate(total)
    f = open(out, "r+b")
    step = (total + n - 1) // n
    lock = threading.Lock()
    done = [0]
    ths = []
    for i in range(n):
        s = i * step
        e = min(total - 1, s + step - 1)
        if s > e:
            break
        t = threading.Thread(target=worker, args=(url, s, e, f, lock, i, done))
        t.start()
        ths.append(t)
    t0 = time.time()
    while any(t.is_alive() for t in ths):
        time.sleep(10)
        el = time.time() - t0
        print(f"  {done[0]/1e6:.1f}/{total/1e6:.1f} MB  {done[0]/1e6/el:.2f} MB/s", flush=True)
    for t in ths:
        t.join()
    f.close()
    print(f"done {done[0]/1e6:.1f} MB in {time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == "__main__":
    main()
