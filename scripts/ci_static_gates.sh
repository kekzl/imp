#!/usr/bin/env bash
# The blocking static gates, in one list, run from two places.
#
# WHY THIS FILE EXISTS. Until 2026-08-21 these ran only as their own CI jobs,
# and branch ruleset 14716423 requires exactly one context: `Build`. Every other
# job is advisory. #1523 merged with `File size` FAILED, #1524 merged over the
# same red gate forty minutes later, and `main` stayed red. A step that blocks
# its own job blocks nothing when the job is not required
# (docs/audit/DEBT_LEDGER_2026_08_21.md section (j)).
#
# So the same list runs as the FIRST step of the `Build` job, where a failure
# makes the one required context red and the PR unmergeable. The named jobs stay
# as they are, because "which gate failed" is worth a check name; they call this
# script with a filter so there is exactly one list rather than two that drift.
#
# WHY NOT `needs:`. Making `Build` depend on the gate jobs would keep the names
# and add no wall time, but it rests on an unverified claim: that a required
# check SKIPPED because its dependency failed blocks a merge rather than being
# treated as satisfied. GitHub's behaviour differs between `needs`-skips and
# path-filter skips, and this campaign's rule is not to build on a claim nobody
# checked. Running inside `Build` needs no such assumption.
#
# WHAT IS DELIBERATELY NOT HERE, and it is a decision rather than an oversight:
#   Lint            apt-installs clang-format and hits the network for upstream
#                   dependency tags. Adding an apt install and a network call to
#                   the one required check trades enforcement for flakiness.
#   Mock API        `pip install -r tests/api/requirements.txt` then pytest. Same
#                   objection as Lint: a network install inside the required
#                   check. It is cheap in CPU and not hermetic, which is the axis
#                   that matters here.
#   clang-tidy      ~1m30, and it is advisory by its own step name.
#   Real API        ~1m50, needs the build artifact, so it cannot run before the
#                   compile it would gate.
#   alloc-interpose costs ~15 minutes and a GPU. It belongs in check-release.sh,
#                   where it already is (stage 9's sibling), not in CI at all.
# Those four stay advisory. Everything in this file is cheap, hermetic and
# deterministic: no apt, no network, no build directory, seconds in total.
set -uo pipefail

# Repo root from this script's own location, not from git. The GitHub Actions
# container runs as a different user than owns the checkout, so `git rev-parse`
# there dies with "detected dubious ownership in repository at /__w/imp/imp" and
# takes the whole required check with it. This needs no git and no safe.directory.
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

# The offline half of the dependency-pin check: cmake/imp-deps.cmake vs the
# Dockerfile ARG defaults, sed/grep over two tracked files, no network. Until
# 2026-09-05 it ran only as `--online` inside the advisory Lint job, so a
# drifted pin merged with a red X (AUDIT_arch_2026 H-1). Lint keeps --online.
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

# Needs a BUILT artifact plus cuobjdump, unlike every other gate here, which is
# source-derived. Skips rather than fails when the build is absent so a fresh
# checkout still gets the rest of the list; CI runs it unconditionally in the
# `Build` job, where both halves exist.
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

# Own group: a file:line citation in a living doc dies the moment a TU is
# split or shrinks, and until 2026-08-26 this surfaced only in CI (the #1782
# scheduler split cost a full CI roundtrip on a roadmap.md citation the
# pre-push never checked). Cheap (<0.5 s), hermetic, covers roadmap.md plus
# every living doc; records (archive/, plans/, audit/) stay excluded.
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
