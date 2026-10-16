#!/bin/bash
# The one pseudopotential of this example: Si, pslibrary 1.0.0 PBEsol ultrasoft, from the QE site,
# checked against the checksum of the file the published numbers were made with.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p pseudo
F=Si.pbesol-n-rrkjus_psl.1.0.0.UPF; SUM=1c533787b17a193082d8817eb50d923d
[ -f "pseudo/$F" ] && echo "$SUM  pseudo/$F" | md5sum -c --quiet - 2>/dev/null \
  || curl -fsSL -m 120 -o "pseudo/$F" "https://pseudopotentials.quantum-espresso.org/upf_files/$F"
echo "$SUM  pseudo/$F" | md5sum -c -
