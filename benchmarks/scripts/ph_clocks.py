#!/usr/bin/env python3
"""Print the clocks of a ph.x output sorted by wall time.

  ph_clocks.py <ph.out> [top-n]         one run
  ph_clocks.py --cmp <ref.out> <new.out> [top-n]   two runs side by side
"""
import re
import sys

CLK = re.compile(r"^\s+(\S+)\s*:\s*(.*?)CPU\s+(.*?)WALL(?:\s*\(\s*(\d+) calls\))?")


def secs(s):
    s = s.strip()
    t = 0.0
    m = re.match(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?$", s)
    if not m:
        return float("nan")
    h, mi, se = m.groups()
    return (int(h) * 3600 if h else 0) + (int(mi) * 60 if mi else 0) + (float(se) if se else 0.0)


def clocks(path):
    # clocks are printed after every q-point and at the end: the last value of each one counts
    out = {}
    for l in open(path, errors="replace").read().splitlines():
        m = CLK.match(l)
        if m:
            out[m.group(1)] = (secs(m.group(3)), secs(m.group(2)), int(m.group(4) or 1))
    return out


def main():
    a = sys.argv[1:]
    if a and a[0] == "--cmp":
        ref, new = clocks(a[1]), clocks(a[2])
        n = int(a[3]) if len(a) > 3 else 40
        names = sorted(set(ref) | set(new), key=lambda k: -max(ref.get(k, (0,))[0], new.get(k, (0,))[0]))
        print(f"{'clock':16s} {'ref wall':>10s} {'new wall':>10s} {'ratio':>7s} {'calls ref/new':>18s}")
        for k in names[:n]:
            r, w = ref.get(k), new.get(k)
            rs = f"{r[0]:10.2f}" if r else " " * 10
            ws = f"{w[0]:10.2f}" if w else " " * 10
            ratio = f"{r[0] / w[0]:7.1f}" if r and w and w[0] > 0 else " " * 7
            calls = f"{r[2] if r else '-'}/{w[2] if w else '-'}"
            print(f"{k:16s} {rs} {ws} {ratio} {calls:>18s}")
        return
    c = clocks(a[0])
    n = int(a[1]) if len(a) > 1 else 45
    print(f"{'clock':16s} {'wall':>10s} {'cpu':>10s} {'calls':>9s}")
    for k, (w, cpu, calls) in sorted(c.items(), key=lambda kv: -kv[1][0])[:n]:
        print(f"{k:16s} {w:10.2f} {cpu:10.2f} {calls:9d}")


if __name__ == "__main__":
    main()
