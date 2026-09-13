#!/usr/bin/env bash
# Blocking static gates run from two places: as CI's own (advisory) jobs, and as the FIRST
# step of the `Build` job, the only required context under branch ruleset 14716423 (#1523/#1524
# merged over red advisory gates because non-required jobs block nothing). Not `needs:` on
# Build: a needs-skip's effect on required-check merging is unverified.
# Deliberately kept advisory: Lint and Mock API (network installs in a required check),
# clang-tidy, Real API (needs the build artifact), alloc-interpose (~15min+GPU, lives in
# check-release.sh). Everything here is cheap, hermetic, deterministic.
set -uo pipefail

# Repo root from this script's own location, not git: the Actions container runs as a
# different user than owns the checkout, so git rev-parse dies with "dubious ownership".
cd "$(dirname "$(readlink -f "$0")")/.."
FAIL=0
run() {  # run <label> <cmd...>
    local label="$1"; shift
    if "$@"; then
        printf '  ok    %s\n' "$label"
    else
        printf '  FAIL  %s\n' "$label"
        FAIL=$((FAIL + 1))
    fi
}

want() {  # no filter = everything
    [ "$#" -eq 0 ] && return 0
    [ "$SELECT_ALL" = "1" ] && return 0
    case " $SELECTED " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

SELECTED="$*"
SELECT_ALL=0
[ -z "$SELECTED" ] && SELECT_ALL=1

if want filesize; then
    echo "== File size =="
    run "hard-review gate + allowlist ceilings" python3 tools/check_filesize.py
    run "that gate still merges #include'd .cu" python3 tools/check_filesize.py --selftest
    run "function bodies over 500 code LOC"     python3 tools/check_function_size.py
    run "that gate still parses what it must"   python3 tools/check_function_size.py --selftest
    run "deterministic-mode sites vs the doc"   python3 tools/check_determinism_sites.py
    run "that gate still catches its drift"     python3 tools/check_determinism_sites.py --selftest
    run "header-inline definitions with no caller" python3 tools/check_dead_inline_accessors.py
    run "FATAL logs that do not stop"           python3 tools/check_log_fatal.py --list
fi

# Own group so its failure carries its own CI check name: both 2026-08-25
# "File size" reds were THIS pin (a new GPU test bumps the unlaned count),
# and the job name sent two readers to the wrong mechanism.
if want lanes; then
    echo "== Test lanes =="
    run "tests that run in no CI lane"          python3 tools/check_test_lanes.py --report
fi

# The container env -> argv translation. Belongs here on this file's own terms:
# no build, no Docker, no network, ~0.2 s. It ran nowhere at all until
# 2026-08-31, which is how IMP_KV_FP8 kept a name whose meaning had inverted.
if want entrypoint; then
    echo "== Entrypoint =="
    run "docker-entrypoint.sh env -> argv"      bash tests/test_entrypoint.sh
fi

# Nothing throws across the C ABI: every `ImpError imp_*()` body in src/api/
# runs under imp::api_guard() or an inline try/catch. Four of 23 had neither
# on 2026-09-05 (AUDIT_arch_2026 G-10).
if want api; then
    echo "== C API =="
    run "ImpError entry points guarded"         python3 tools/check_api_guard.py
    run "that gate still classifies its cases"  python3 tools/check_api_guard.py --selftest
fi

# Offline half of the dependency-pin check (cmake/imp-deps.cmake vs Dockerfile ARG defaults,
# textual, no network). Previously ran only as --online inside the advisory Lint job, so a
# drifted pin could merge (AUDIT_arch_2026 H-1). Lint still runs --online.
if want deps; then
    echo "== Dependency pins =="
    run "cmake/imp-deps.cmake vs Dockerfile (offline)" bash scripts/check_dep_pins.sh
    run "that gate still names its violations"        bash scripts/check_dep_pins.sh --selftest
fi

if want alloc; then
    echo "== Alloc sites =="
    run "I1 allowlist gate"                     python3 tools/check_alloc_sites.py
    run "allocate/free API pairing"             python3 tools/check_alloc_pairs.py
    # Lazy device statics re-arm across ~Engine / imp_gpu_release (the
    # IMP_REGISTER_CUDA_STATIC_RESET convention). Six TUs had none on
    # 2026-09-05, one of them a use-after-free on the default model-swap path.
    run "lazy device statics re-arm"            python3 tools/check_static_reset.py
    run "that gate still classifies its cases"  python3 tools/check_static_reset.py --selftest
fi

# Needs a BUILT artifact + cuobjdump (unlike every other gate here, which is source-derived).
# Skips (not fails) when the build is absent, so a fresh checkout still gets the rest of the list.
if want kernels; then
    echo "== Kernel resources =="
    KRES_LIB=""
    [ -f build/libimp.a ] && KRES_LIB=build/libimp.a
    [ -z "$KRES_LIB" ] && [ -f build-dev/libimp.a ] && KRES_LIB=build-dev/libimp.a
    if [ -z "$KRES_LIB" ]; then
        echo "  (skipped: no libimp.a — run 'make dev' or 'make build' first)"
    elif ! command -v cuobjdump >/dev/null 2>&1 && ! docker image inspect imp:builder >/dev/null 2>&1; then
        echo "  (skipped: no cuobjdump and no imp:builder image)"
    else
        run "registers + local frame vs the pin" make -s kernel-resources
    fi
fi

if want launchguards; then
    echo "== Launch guards =="
    run "post-launch check gate"                python3 tools/check_launch_guards.py
fi

if want docs; then
    echo "== Docs =="
    run "generated perf blocks match baseline"  python3 scripts/sync_docs.py --check
    run "doc lint (layers, provenance, links)"  python3 scripts/docs_lint.py
    # imp.conf.example is the only key catalogue; 31 of 223 bound keys were
    # missing from it on 2026-09-05 (AUDIT_arch_2026 J-2). CHANGELOG entries:
    # 19 of the last 26 broke the 3-line rule nothing checked (J-8).
    run "imp.conf.example lists every bound key"   python3 tools/check_config_keys.py
    run "that gate still sees a missing key"       python3 tools/check_config_keys.py --selftest
    run "CHANGELOG [Unreleased] entries <= 3 lines" python3 tools/check_changelog_form.py
    run "that gate still counts a long entry"      python3 tools/check_changelog_form.py --selftest
fi

# A file:line citation in a living doc dies the moment a TU is split or shrinks (#1782
# scheduler split cost a CI roundtrip on a roadmap.md citation pre-push never checked).
# Cheap (<0.5s), covers roadmap.md plus every living doc; archive/plans/audit stay excluded.
if want layering; then
    echo "== Layering =="
    # Backward #include edges between src/ layers against tools/layering_pins.txt.
    # 88 lines on 2026-09-05, 24 after AUDIT_arch_2026 dispatch #14; the dead
    # runtime/config.h include in exec/ had come back once (#1388) with no gate.
    run "backward layer includes pinned"        python3 tools/check_layering.py
    run "that gate still classifies its cases"  python3 tools/check_layering.py --selftest
fi

if want citations; then
    echo "== Doc citations =="
    run "file:line citations in living docs"    python3 scripts/check_doc_citations.py .
fi

if want hygiene; then
    echo "== Release hygiene =="
    run "check-release.sh without the GPU gate" env SKIP_VERIFY=1 bash scripts/check-release.sh
fi

echo
if [ "$FAIL" -ne 0 ]; then
    echo "ci-static-gates: $FAIL gate(s) failed"
    exit 1
fi
echo "ci-static-gates: all selected gates passed"
