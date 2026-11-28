#!/bin/bash
# Rate objective on a well that the schedule defines after the first report
# step. MODEL_1D_DEBUG with the producer's WELSPECS/COMPDAT/WCONPROD moved
# behind the first TSTEP: during the first report step the producer is not in
# the well state, so its rate terms must count as q = 0 there (as in the
# summary) instead of aborting the sweep.
#
# Checks: the objective-only replay and the gradient sweep run and agree on J,
# J is nonzero, and J equals the same objective restricted to the second
# report step (the first one contributes nothing).
#
# Usage:
#   run-adjoint-late-well-test.sh <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [rel-tol]

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <inputfiles-dir> <deck-name> <output-dir> [rel-tol]}
INPUTDIR=${2:?missing inputfiles dir}
DECKNAME=${3:?missing deck name}
OUTDIR=${4:?missing output dir}
RELTOL=${5:-1e-12}

WELL=Prod
OBJECTIVE="rate:${WELL}:oil"
COMMON="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false"

rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"
INPUT=${OUTDIR}/input
cp -r "${INPUTDIR}" "${INPUT}"

welspecs=$(grep "^${WELL} " "${INPUT}/welspecs.txt")
compdat=$(grep "^${WELL} " "${INPUT}/compdat.txt")
wconprod=$(grep "^${WELL} " "${INPUT}/wconprodstep_1.txt")
step=$(awk 'NR==2 {print $1}' "${INPUT}/tstepstep_1.txt")
if [ -z "${welspecs}" ] || [ -z "${compdat}" ] || [ -z "${wconprod}" ] || [ -z "${step}" ]; then
    echo "FAIL: unexpected layout of the ${DECKNAME} include files"
    exit 1
fi
# -i with a suffix: plain -i is GNU sed only
sed -i.bak "/^${WELL} /d" "${INPUT}/welspecs.txt" "${INPUT}/compdat.txt" \
    && rm -f "${INPUT}"/*.bak
if grep -q "^${WELL} " "${INPUT}/welspecs.txt" "${INPUT}/compdat.txt"; then
    echo "FAIL: could not move ${WELL} out of the initial WELSPECS/COMPDAT"
    exit 1
fi
printf 'WCONPROD\n/\n' > "${INPUT}/wconprodstep_1.txt"
printf 'TSTEP\n%s\n/\n\nWELSPECS\n%s\n/\n\nCOMPDAT\n%s\n/\n\nWCONPROD\n%s\n/\n\nTSTEP\n%s\n/\n' \
    "${step}" "${welspecs}" "${compdat}" "${wconprod}" "${step}" > "${INPUT}/tstepstep_1.txt"

DECK=${INPUT}/${DECKNAME}.DATA
RUN=${OUTDIR}/run
${FLOW} "${DECK}" --output-dir="${RUN}" ${COMMON} --adjoint-save=true \
    > "${OUTDIR}/fwd.log" 2>&1 || { echo "FAIL: forward run"; exit 1; }

objective() { # <spec> <log> -> prints J
    ${FLOW} "${DECK}" --output-dir="${RUN}" ${COMMON} --adjoint-mode=objective \
        --adjoint-objective="$1" > "$2" 2>&1 || { echo FAIL; return; }
    grep -o "Adjoint objective J = [0-9eE.+-]*" "$2" | awk '{print $5}'
}

J_ALL=$(objective "${OBJECTIVE}" "${OUTDIR}/obj_all.log")
J_LATE=$(objective "${OBJECTIVE}:t0=$(awk "BEGIN {print 1.5 * ${step}}")" "${OUTDIR}/obj_late.log")

${FLOW} "${DECK}" --output-dir="${RUN}" ${COMMON} --adjoint-mode=gradient \
    --adjoint-objective="${OBJECTIVE}" > "${OUTDIR}/grad.log" 2>&1 \
    || { echo "FAIL: gradient sweep"; exit 1; }
J_GRAD=$(grep -o "Adjoint sweep finished: J = [0-9eE.+-]*" "${OUTDIR}/grad.log" | awk '{print $6}')

echo "J (whole run)          = ${J_ALL}"
echo "J (second report step) = ${J_LATE}"
echo "J (gradient sweep)     = ${J_GRAD}"

python3 - "${J_ALL}" "${J_LATE}" "${J_GRAD}" "${RELTOL}" << 'EOF'
import sys
try:
    j_all, j_late, j_grad, tol = (float(v) for v in sys.argv[1:])
except ValueError:
    sys.exit("FAIL: could not read the objective values")
if j_all == 0.0:
    sys.exit("FAIL: the objective is zero")
for name, value in (("second report step", j_late), ("gradient sweep", j_grad)):
    if abs(value - j_all) > tol * abs(j_all):
        sys.exit(f"FAIL: J of the {name} differs from J of the whole run")
print("adjoint late-well objective test: PASS")
EOF
