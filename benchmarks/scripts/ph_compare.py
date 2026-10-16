#!/usr/bin/env python3
"""Compare two ph.x runs (directories written by run_ph.sh).

  ph_compare.py <ref-dir> <new-dir>              one case
  ph_compare.py --suite <ref-root> <new-root> [case ...]   table over cases

Reported: wall time (PHONON clock) and speedup, number of DFPT iterations, largest difference of
the phonon frequencies (cm-1), of the dynamical-matrix elements (relative to the largest element),
of the dielectric tensor and of the effective charges.
"""
import glob
import os
import re
import sys

FREQ = re.compile(r"freq \(\s*\d+\)\s*=\s*(-?[\d.]+)\s*\[THz\]\s*=\s*(-?[\d.]+)\s*\[cm-1\]")
NUM = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"


def wall(text):
    m = re.findall(r"^\s+PHONON\s*:.*CPU\s+(.*?)WALL", text, re.M)
    if not m:
        return None
    s = m[-1].strip()
    mm = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?$", s)
    h, mi, se = mm.groups()
    return (int(h) * 3600 if h else 0) + (int(mi) * 60 if mi else 0) + (float(se) if se else 0.0)


def parse(d):
    out = {"dir": d, "ok": False}
    p = os.path.join(d, "ph.out")
    if not os.path.exists(p):
        return out
    t = open(p, errors="replace").read()
    out["ok"] = "JOB DONE" in t
    out["wall"] = wall(t)
    out["iters"] = len(re.findall(r"^\s+iter #", t, re.M))
    out["freq"] = [float(x[1]) for x in FREQ.findall(t)]
    out["notconv"] = len(re.findall(r"root not converged|No convergence has been achieved", t))
    eps = []
    for blk in re.findall(r"Dielectric constant in cartesian axis\s*\n((?:\s*\n)*(?:\s*\(.*\)\s*\n){3})", t):
        eps += [float(x) for x in re.findall(NUM, blk)]
    out["eps"] = eps
    zs = []
    for blk in re.findall(r"Effective charges \(d Force / dE\) in cartesian axis(?: without acoustic sum rule applied \(asr\))?\s*\n(.*?)(?:Effective charges Sum|\n\s*\n\s*\n)", t, re.S):
        for line in blk.splitlines():
            if re.match(r"\s*E[xyz]", line):
                zs += [float(x) for x in re.findall(NUM, line)]
    out["zeu"] = zs
    dyn = []
    for f in sorted(glob.glob(os.path.join(d, "dyn*"))):
        if f.endswith(".xml") or f.endswith("dyn0"):
            continue
        txt = open(f, errors="replace").read()
        for blk in re.findall(r"Dynamical\s+Matrix in cartesian axes(.*?)(?:Dielectric Tensor|Diagonalizing the dynamical matrix|Dynamical\s+Matrix in cartesian axes)", txt, re.S):
            for line in blk.splitlines():
                v = re.findall(NUM, line)
                if len(v) == 6:
                    dyn += [float(x) for x in v]
    out["dyn"] = dyn
    return out


def maxdiff(a, b):
    if not a and not b:
        return None
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def compare(ref, new):
    r, n = parse(ref), parse(new)
    res = {"ok": r["ok"] and n["ok"], "ref": r, "new": n}
    if not res["ok"]:
        return res
    res["dfreq"] = maxdiff(r["freq"], n["freq"])
    dd = maxdiff(r["dyn"], n["dyn"])
    scale = max((abs(x) for x in r["dyn"]), default=1.0) or 1.0
    res["ddyn"] = None if dd is None else dd / scale
    res["deps"] = maxdiff(r["eps"], n["eps"])
    res["dzeu"] = maxdiff(r["zeu"], n["zeu"])
    return res


def fmt(x, f="{:.1e}"):
    return "-" if x is None else f.format(x)


def main():
    a = sys.argv[1:]
    if a and a[0] == "--suite":
        ref_root, new_root = a[1], a[2]
        cases = a[3:] or sorted(c for c in os.listdir(new_root) if os.path.isdir(os.path.join(new_root, c)))
        print(f"{'case':24s} {'ref s':>8s} {'new s':>8s} {'speedup':>8s} {'it ref/new':>11s} "
              f"{'dfreq cm-1':>10s} {'ddyn rel':>9s} {'deps':>8s} {'dZ*':>8s}")
        tr = tn = 0.0
        import math
        logs = []
        for c in cases:
            res = compare(os.path.join(ref_root, c), os.path.join(new_root, c))
            r, n = res["ref"], res["new"]
            if not res["ok"]:
                print(f"{c:24s} {'ref ' + ('ok' if r['ok'] else 'MISSING'):>8s} {'new ' + ('ok' if n['ok'] else 'FAILED'):>12s}")
                continue
            sp = r["wall"] / n["wall"]
            tr += r["wall"]; tn += n["wall"]; logs.append(math.log(sp))
            flag = " NOTCONV" if n["notconv"] else ""
            print(f"{c:24s} {r['wall']:8.1f} {n['wall']:8.1f} {sp:7.1f}x {r['iters']:5d}/{n['iters']:<5d} "
                  f"{fmt(res['dfreq']):>10s} {fmt(res['ddyn']):>9s} {fmt(res['deps']):>8s} {fmt(res['dzeu']):>8s}{flag}")
        if logs:
            print(f"{'total':24s} {tr:8.1f} {tn:8.1f} {tr / tn:7.1f}x   geomean {math.exp(sum(logs) / len(logs)):.1f}x")
        return
    res = compare(a[0], a[1])
    r, n = res["ref"], res["new"]
    if not res["ok"]:
        print(f"ref ok={r['ok']} new ok={n['ok']}")
        sys.exit(1)
    print(f"wall {r['wall']:.2f} -> {n['wall']:.2f} s ({r['wall'] / n['wall']:.2f}x), iterations {r['iters']} -> {n['iters']}")
    print(f"frequencies: {len(n['freq'])}, max diff {fmt(res['dfreq'])} cm-1; dyn elements: {len(n['dyn'])}, "
          f"max rel diff {fmt(res['ddyn'])}; eps diff {fmt(res['deps'])}; Z* diff {fmt(res['dzeu'])}")


if __name__ == "__main__":
    main()
