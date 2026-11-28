#!/bin/bash
# Rate objective on a well that the schedule defines after the first report
# step. MODEL_1D_DEBUG with the producer's WELSPECS/COMPDAT/WCONPROD moved
# behind the first TSTEP: during the first report step the producer is not in
# the well state, so its rate terms must count as q = 0 there (as in the
# summary) instead of aborting the sweep.
#
# Checks, for the oil-rate objective: the objective-only replay and the
# gradient sweep run and agree on J, J is nonzero, J equals the same objective
# restricted to the second report step (the first one contributes nothing),
# and the pore-volume gradient matches finite differences. The bhp objective
# and a matchref term, whose reference run has the producer from the start,
# must run through both modes and agree on a nonzero J as well.
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

HERE=$(cd "$(dirname "$0")" && pwd)

rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"
INPUT=${OUTDIR}/input
cp -r "${INPUTDIR}" "${INPUT}"

# Reference for the matchref term: the deck as it is, producer from the start
REF=${OUTDIR}/reference
${FLOW} "${INPUTDIR}/${DECKNAME}.DATA" --output-dir="${REF}" ${COMMON} \
    > "${OUTDIR}/reference.log" 2>&1 || { echo "FAIL: reference run"; exit 1; }

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

gradient() { # <spec> <log> -> prints J of the gradient sweep
    ${FLOW} "${DECK}" --output-dir="${RUN}" ${COMMON} --adjoint-mode=gradient \
        --adjoint-objective="$1" > "$2" 2>&1 || { echo FAIL; return; }
    grep -o "Adjoint sweep finished: J = [0-9eE.+-]*" "$2" | awk '{print $6}'
}

same() { # <label> <J> <J> ...: all nonzero and equal to the first within RELTOL
    python3 - "${RELTOL}" "$@" << 'EOF'
import sys
tol, label, values = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
try:
    values = [float(v) for v in values]
except ValueError:
    sys.exit(f"FAIL: {label}: a run failed or printed no objective value")
if values[0] == 0.0:
    sys.exit(f"FAIL: {label}: the objective is zero")
if any(abs(v - values[0]) > tol * abs(values[0]) for v in values):
    sys.exit(f"FAIL: {label}: the objective values differ: {values}")
EOF
}

J_ALL=$(objective "${OBJECTIVE}" "${OUTDIR}/obj_all.log")
J_LATE=$(objective "${OBJECTIVE}:t0=$(awk "BEGIN {print 1.5 * ${step}}")" "${OUTDIR}/obj_late.log")
J_GRAD=$(gradient "${OBJECTIVE}" "${OUTDIR}/grad.log")
echo "J (whole run)          = ${J_ALL}"
echo "J (second report step) = ${J_LATE}"
echo "J (gradient sweep)     = ${J_GRAD}"
same "${OBJECTIVE}" "${J_ALL}" "${J_LATE}" "${J_GRAD}" || exit 1

for spec in "bhp:${WELL}" "matchref:${WELL}:oil:${REF}/${DECKNAME}"; do
    kind=${spec%%:*}
    J_OBJ=$(objective "${spec}" "${OUTDIR}/obj_${kind}.log")
    J_GRAD=$(gradient "${spec}" "${OUTDIR}/grad_${kind}.log")
    echo "J (${kind}) = ${J_OBJ} (objective replay), ${J_GRAD} (gradient sweep)"
    same "${kind}" "${J_OBJ}" "${J_GRAD}" || exit 1
done

LC_ALL=C "${HERE}/run-adjoint-fd-test.sh" "${FLOW}" "${INPUT}" "${DECKNAME}" "${OUTDIR}/fd" \
    1e-6 1e-4 "${OBJECTIVE}" > "${OUTDIR}/fd.log" 2>&1 \
    || { tail -5 "${OUTDIR}/fd.log"; echo "FAIL: gradient against finite differences"; exit 1; }
grep "^cell" "${OUTDIR}/fd.log"

echo "adjoint late-well objective test: PASS"
