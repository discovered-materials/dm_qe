#!/bin/bash
# Fetch the pseudopotentials of the benchmark cases into benchmarks/ph/pseudo from their public
# sources and check them against the checksums of the files the published numbers were made with:
#   PseudoDojo NC SR v0.4 PBE "standard" (www.pseudo-dojo.org): C Mg Mo N Ni O S Si -> <El>.dojo-sr-04-pbe-std.upf
#   pslibrary 1.0.0 PBEsol PAW / ultrasoft (pseudopotentials.quantum-espresso.org): Al C N PAW, Si ultrasoft
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); P=$(cd "$HERE/.." && pwd)/ph/pseudo
mkdir -p "$P"; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
MD5='fed048700f41f568201cacb08e201ac8  Al.pbesol-n-kjpaw_psl.1.0.0.UPF
f9b2fe17d1f478429498b05d17159f9e  C.pbesol-n-kjpaw_psl.1.0.0.UPF
48ecc4ab5d29f7d3bdae09f74cf45e2a  N.pbesol-n-kjpaw_psl.1.0.0.UPF
1c533787b17a193082d8817eb50d923d  Si.pbesol-n-rrkjus_psl.1.0.0.UPF
34a24e64c0a39f27c6c36b90a16ac686  C.dojo-sr-04-pbe-std.upf
4c04d8c6e0476e83eaecaf72fa8acc4d  Mg.dojo-sr-04-pbe-std.upf
7ca7a9feba7bf08d88d14bc7fbf60c1e  Mo.dojo-sr-04-pbe-std.upf
d34365f73bb0e2ee98afc20ec3e87676  N.dojo-sr-04-pbe-std.upf
2b0a79d100e227970886d6963e34175b  Ni.dojo-sr-04-pbe-std.upf
3e3f909f7470e8921608ab6b1f57bbae  O.dojo-sr-04-pbe-std.upf
6027e7704d67ae28863b2247926ce6f3  S.dojo-sr-04-pbe-std.upf
02fab3f35e82123ef5bf1cb05d5b1a5e  Si.dojo-sr-04-pbe-std.upf'
if ! ( cd "$P" && echo "$MD5" | md5sum -c --quiet - ) > /dev/null 2>&1; then
  echo "== PseudoDojo nc-sr-04 PBE standard"
  curl -fsSL -m 300 -o "$TMP/sr.tgz" https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz
  mkdir -p "$TMP/sr" && tar -xzf "$TMP/sr.tgz" -C "$TMP/sr"
  for el in C Mg Mo N Ni O S Si; do
    src=$(find "$TMP/sr" -name "$el.upf" | head -1); [ -n "$src" ] || { echo "no $el.upf in the PseudoDojo archive" >&2; exit 1; }
    cp "$src" "$P/$el.dojo-sr-04-pbe-std.upf"
  done
  echo "== pslibrary 1.0.0 PBEsol"
  for f in Al.pbesol-n-kjpaw_psl.1.0.0.UPF C.pbesol-n-kjpaw_psl.1.0.0.UPF N.pbesol-n-kjpaw_psl.1.0.0.UPF Si.pbesol-n-rrkjus_psl.1.0.0.UPF; do
    curl -fsSL -m 120 -o "$P/$f" "https://pseudopotentials.quantum-espresso.org/upf_files/$f"
  done
fi
( cd "$P" && echo "$MD5" | md5sum -c --quiet - ) && echo "$P: 12 pseudopotentials, checksums as published"
