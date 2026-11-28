#!/bin/bash
# FD verification of the RESV well-control derivatives (AdjointResvTerm).
#
# SPE1CASE1 with the producer on history-mode RESV control (WCONHIST RESV,
# history ORAT 20000 STB/d and GRAT 25400 MSCF/d, BHP limit 1000 psi kept via
# WELTARG) and DRSDT removed (its own cross-step gap is a separate, documented
# term). A RESV control converts surface rates to reservoir voidage with
# coefficients, and a WCONHIST target, evaluated at field averages of the
# previous substep's state, and the coefficients also depend on the well's own
# rates through the Rs/Rv clamp; the simulator's Jacobian carries neither
# dependence. Perturb all 300 PORO uniformly (pvmult = 1 +- eps) and compare
# the central difference of the oil-rate objective with sum_i dJ/dpvmult_i:
#   --adjoint-resv-derivatives=true  -> must agree to RELTOL (asserted)
#   --adjoint-resv-derivatives=false -> deviation printed for reference.
#
# usage: run-adjoint-fd-resv-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir> [eps] [reltol]
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
EPS=${4:-1e-5}
RELTOL=${5:-1e-5}
rm -rf "$OUT"
mkdir -p "$OUT"

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-10 --tolerance-cnv-relaxed=1e-10 \
 --tolerance-mb=1e-12 --tolerance-mb-relaxed=1e-12 \
 --tolerance-wells=1e-10 --relaxed-well-flow-tol=1e-10"
OBJ=rate:PROD:oil

python3 - "$DECK" "$OUT" "$EPS" << 'EOF'
import sys, os
src, out, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
t = open(src).read()
assert '300*0.3' in t, "expected inline '300*0.3' PORO in SPE1CASE1"
assert 'DRSDT\n 0 /\n' in t, 'unexpected DRSDT layout'
t = t.replace('DRSDT\n 0 /\n', '')
prod = "\t'PROD' 'OPEN' 'ORAT' 20000 4* 1000 /"
assert prod in t, 'unexpected WCONPROD layout'
t = t.replace("WCONPROD", "WCONHIST", 1)
t = t.replace(prod, "\t'PROD' 'OPEN' 'RESV' 20000 0 25400 /\n/\n\nWELTARG\n\t'PROD' BHP 1000 /")
for tag, sgn in (('base', 0.0), ('plus', eps), ('minus', -eps)):
    os.makedirs(f'{out}/deck_{tag}', exist_ok=True)
    open(f'{out}/deck_{tag}/SPE1RESV.DATA', 'w').write(
        t.replace('300*0.3', f'300*{0.3 * (1.0 + sgn):.16f}'))
EOF

getJ() { grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

for tag in base plus minus; do
    $FLOW "$OUT/deck_$tag/SPE1RESV.DATA" --output-dir="$OUT/run_$tag" $STRICT \
        --adjoint-save=true --adjoint-file="$OUT/run_$tag/a" > "$OUT/fwd_$tag.log" 2>&1 \
        || { echo "FAIL: forward $tag"; exit 1; }
done
for s in plus minus; do
    $FLOW "$OUT/deck_$s/SPE1RESV.DATA" --output-dir="$OUT/run_$s" $STRICT \
        --adjoint-mode=objective --adjoint-file="$OUT/run_$s/a" \
        --adjoint-objective=$OBJ > "$OUT/obj_$s.log" 2>&1
done
JP=$(getJ "$OUT/obj_plus.log"); JM=$(getJ "$OUT/obj_minus.log")

dev() { # <true|false>: relative FD-vs-adjoint deviation of the uniform PV direction
    local d="$OUT/grad_$1"
    $FLOW "$OUT/deck_base/SPE1RESV.DATA" --output-dir="$d" $STRICT \
        --adjoint-mode=gradient --adjoint-file="$OUT/run_base/a" \
        --adjoint-objective=$OBJ --adjoint-resv-derivatives=$1 \
        --adjoint-linear-solver-reduction=1e-16 > "$d.log" 2>&1
    python3 -c "
g = [float(l.split()[0]) for l in open('$d/SPE1RESV.ADJOINT_GRADIENTS_PV.txt').read().splitlines()[1:]]
adj = sum(g); fd = ($JP - $JM) / (2*$EPS)
print(repr((fd - adj) / fd))"
}

DEV_ON=$(dev true)
DEV_OFF=$(dev false)
grep -h "Adjoint RESV controls" "$OUT/grad_true.log" | head -1
printf "RESV derivatives on : rel.dev = %s   (must be < %s)\n" "$DEV_ON" "$RELTOL"
printf "RESV derivatives off: rel.dev = %s   (reference)\n" "$DEV_OFF"
python3 -c "import sys; sys.exit(0 if abs($DEV_ON) < $RELTOL else 1)" \
    && { echo "adjoint RESV control FD test: PASS"; exit 0; }
echo "adjoint RESV control FD test: FAIL"; exit 1
