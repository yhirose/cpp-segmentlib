#!/usr/bin/env bash
# Benchmarks segmentlib against KyTea and Vaporetto on word segmentation.
#
# Methodology (see bench/README.md):
#   1. All three run the SAME model (the KyTea jp model, converted for
#      Vaporetto), so we compare speed of (nearly) the same computation.
#   2. Correctness gate: outputs are diffed first. segmentlib is byte-identical
#      to KyTea; Vaporetto's divergence rate is reported.
#   3. Pure inference speed is measured IN-PROCESS (model loaded once, the
#      tokenize loop timed, model load and I/O excluded) — the only way to
#      isolate inference. A CLI wall-clock comparison, dominated by load and I/O
#      here, cannot. End-to-end CLI wall time is reported separately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KMODEL="${KMODEL:-models/kytea/jp-0.4.7-5.mod}"
VMODEL="${VMODEL:-models/kytea/jp-0.4.7-5.vaporetto.zst}"
OURS="${OURS:-build-release/src/cli/segmenter}"
KYTEA="${KYTEA:-kytea}"
VAPORETTO="${VAPORETTO:-bench/.vendor/vaporetto-predict}"
CORPUS="${CORPUS:-bench/corpus/aozora.txt}"
BENCH_SEGMENT="${BENCH_SEGMENT:-build-release/bench/bench_segment}"
BENCH_KYTEA="${BENCH_KYTEA:-bench/.vendor/bench_kytea}"
ITERS="${ITERS:-7}"

for f in "$KMODEL" "$VMODEL" "$OURS" "$VAPORETTO" "$CORPUS" "$BENCH_SEGMENT"; do
    [[ -e "$f" ]] || { echo "missing: $f  (run bench/setup.sh first)" >&2; exit 1; }
done

CODEPOINTS=$(python3 -c "print(len(open('$CORPUS',encoding='utf-8').read()))")
LINES=$(wc -l < "$CORPUS" | tr -d ' ')

echo "# segmentlib segmentation benchmark"
echo
echo "corpus: $CORPUS ($LINES lines, $CODEPOINTS codepoints)"
echo "cpu:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo

# ---- correctness gate ---------------------------------------------------
echo "## Correctness gate (same model, outputs compared)"
echo
K_OUT="$(mktemp)"; O_OUT="$(mktemp)"; V_OUT="$(mktemp)"
trap 'rm -f "$K_OUT" "$O_OUT" "$V_OUT"' EXIT
"$KYTEA" -model "$KMODEL" -notags < "$CORPUS" > "$K_OUT" 2>/dev/null
"$OURS" predict --notags --model "$KMODEL" < "$CORPUS" > "$O_OUT" 2>/dev/null
"$VAPORETTO" --model "$VMODEL" < "$CORPUS" > "$V_OUT" 2>/dev/null
gate() {
    local total diff pct
    total=$(wc -l < "$3" | tr -d ' ')
    diff=$(paste -d '\t' "$2" "$3" | awk -F '\t' '$1!=$2' | wc -l | tr -d ' ')
    pct=$(python3 -c "print(f'{100*(1-$diff/$total):.2f}')")
    printf "%-22s %5s / %s lines differ  (%s%% identical)\n" "$1" "$diff" "$total" "$pct"
}
gate "segmentlib vs KyTea:" "$O_OUT" "$K_OUT"
gate "Vaporetto  vs KyTea:" "$V_OUT" "$K_OUT"
echo

# ---- pure inference (in-process) ---------------------------------------
echo "## Pure inference speed (in-process; model load + I/O excluded)"
echo

parse_speed() { grep -oE '[0-9.]+ M chars/sec' | grep -oE '[0-9.]+' | head -1; }
parse_load()  { grep -oE 'model load:[[:space:]]*[0-9]+' | grep -oE '[0-9]+' | head -1; }

OURS_OUT=$("$BENCH_SEGMENT" "$KMODEL" "$CORPUS" "$ITERS")
OURS_MCPS=$(echo "$OURS_OUT" | parse_speed); OURS_LOAD=$(echo "$OURS_OUT" | parse_load)

if [[ -x "$BENCH_KYTEA" ]]; then
    K_OUT2=$(DYLD_LIBRARY_PATH=/opt/homebrew/lib "$BENCH_KYTEA" "$KMODEL" "$CORPUS" "$ITERS")
    K_MCPS=$(echo "$K_OUT2" | parse_speed); K_LOAD=$(echo "$K_OUT2" | parse_load)
else
    K_MCPS="n/a"; K_LOAD="n/a"
fi

# Vaporetto reports its own tokenization time ("Elapsed") on stderr; take the
# best of a few runs. Its load is the wall time minus that.
V_BEST=1e9; V_LOAD="n/a"
for _ in 1 2 3 4 5; do
    ELAPSED=$("$VAPORETTO" --model "$VMODEL" < "$CORPUS" 2>&1 >/dev/null \
        | grep -oE 'Elapsed:[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' || true)
    [[ -n "$ELAPSED" ]] && V_BEST=$(python3 -c "print(min($V_BEST,$ELAPSED))")
done
V_MCPS=$(python3 -c "print(f'{$CODEPOINTS/$V_BEST/1e6:.2f}')")

python3 - "$OURS_MCPS" "$OURS_LOAD" "$K_MCPS" "$K_LOAD" "$V_MCPS" <<'PY'
import sys
o_mcps, o_load, k_mcps, k_load, v_mcps = sys.argv[1:6]
def ratio(x):
    try: return f"{float(x)/float(k_mcps):.2f}x"
    except: return "-"
print(f"| {'tool':<12} | {'load (ms)':>9} | {'M chars/sec':>11} | {'vs KyTea':>8} |")
print(f"|{'-'*14}|{'-'*11}|{'-'*13}|{'-'*10}|")
print(f"| {'segmentlib':<12} | {o_load:>9} | {o_mcps:>11} | {ratio(o_mcps):>8} |")
print(f"| {'KyTea':<12} | {k_load:>9} | {k_mcps:>11} | {ratio(k_mcps):>8} |")
print(f"| {'Vaporetto':<12} | {'~load*':>9} | {v_mcps:>11} | {ratio(v_mcps):>8} |")
PY
echo
echo "load = time to read the model and build in-memory structures."
echo "* Vaporetto builds a double-array automaton at load; its load (several"
echo "  seconds) is much larger than KyTea's/segmentlib's but is a one-time cost."
