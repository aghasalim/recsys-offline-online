#!/usr/bin/env bash
# Recompute the published numbers in seven other languages and require agreement.
#
# Every number in the README came out of one pandas and numpy pipeline in
# src/roo/. Nothing downstream could catch an error in it, because everything
# downstream reads the same JSON the pipeline wrote. The tests checked that the
# code ran, not that it was right.
#
# So each implementation here rebuilds a different published table from a rawer
# layer of the same data, and one of them checks the README itself against the
# files. An arithmetic mistake would have to be made identically in several
# languages to survive.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them installed. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQLite has no way to exit non-zero on its own, so the script ends with a
# single RESULT line carrying the failure count and the number of scenarios it
# paired. Anything but a clean run of all seven is a failure here.
check_sql () {
    local out
    out=$(sqlite3 :memory: ".read verify/gate.sql" 2>&1) || { printf '%s\n' "$out"; return 1; }
    printf '%s\n' "$out"
    grep -qx 'RESULT 0 7' <<<"$out"
}

check_c () {
    local bin="${TMPDIR:-/tmp}/roo_groundtruth"
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$bin" verify/groundtruth.c -lm || return 1
    "$bin" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/thresholds && cargo run --release --quiet -- "$root" ); }

run "C, the ground truth table"        cc      check_c
run "SQL, the gate table"              sqlite3 check_sql
run "Go, file structure and the gate"  go      check_go
run "R, the headline inference"        Rscript Rscript verify/verify.R "$root"
run "Ruby, the estimator tables"       ruby    ruby verify/estimators.rb "$root"
run "JavaScript, the README itself"    node    node verify/readme.js "$root"
run "Rust, the gate's thresholds"      cargo   check_rust

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
