#!/usr/bin/env python3
"""Compare the failures of testcode runs with the known-failure list.

  compare_failures.py --known known-failures.txt [--summary out.md] <testcode log> ...

A test is "<directory>/<input>" as testcode names it ("ph_metal - al.elph.in (arg(s): 2): **FAILED**."
becomes ph_metal/al.elph.in). Exit 1 when a test fails that is not on the list, or when a log has no
"All done" summary (testcode did not finish). Known failures that now pass are reported, not an error:
remove them from the list.
"""
import argparse, re, sys

LINE = re.compile(r"^(\S+) - (\S+) \(arg\(s\): \d+\): (Passed|\*\*FAILED\*\*|Skipped|WARNING)", re.M)
DONE = re.compile(r"^All done\..*$", re.M)

def known_set(path):
    out = set()
    try:
        for l in open(path):
            l = l.split("#", 1)[0].strip()
            if l:
                out.add(l)
    except FileNotFoundError:
        pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", required=True)
    ap.add_argument("--summary")
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    known = known_set(a.known)
    passed, failed, skipped, unfinished, totals = [], [], [], [], []
    for p in a.logs:
        if p.endswith(("build-gpu.log", "build-cpu.log", "prolog.log")):
            continue
        t = open(p, errors="replace").read()
        for d, inp, st in LINE.findall(t):
            key = f"{d}/{inp}"
            (passed if st == "Passed" else failed if st == "**FAILED**" else skipped).append(key)
        m = DONE.findall(t)
        if m:
            totals.append(f"{p.rsplit('/',1)[-1]}: {m[-1]}")
        else:
            unfinished.append(p.rsplit("/", 1)[-1])
    failed = sorted(set(failed))
    new = [k for k in failed if k not in known]
    expected = [k for k in failed if k in known]
    fixed = sorted(k for k in known if k in set(passed))
    lines = ["## Test-suite", ""]
    lines += [f"- {t}" for t in totals]
    lines.append(f"- {len(set(passed))} passed, {len(failed)} failed ({len(expected)} known), {len(set(skipped))} skipped")
    if new:
        lines += ["", "**New failures** (not in known-failures.txt):"] + [f"- `{k}`" for k in new]
    if expected:
        lines += ["", "Known failures, still failing:"] + [f"- `{k}`" for k in expected]
    if fixed:
        lines += ["", "Known failures that pass now (remove them from the list):"] + [f"- `{k}`" for k in fixed]
    if unfinished:
        lines += ["", "**Runs without a final summary** (testcode did not finish):"] + [f"- {u}" for u in unfinished]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.summary:
        open(a.summary, "w").write(text)
    return 1 if (new or unfinished) else 0

if __name__ == "__main__":
    sys.exit(main())
