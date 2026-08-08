#!/usr/bin/env python3
"""Mutation-testing harness for imp.

For each mutant in the catalogue: inject a semantically meaningful fault into
one production source file, rebuild incrementally, run the test suite, and
record whether any test noticed. A surviving mutant is a confirmed test gap.

Design notes that matter:

* **Revert is never `git checkout`.** The original bytes are captured before
  the edit and written back afterwards, and a `git status --porcelain` check
  runs after every mutant. `git checkout -- <file>` would also discard any
  unrelated uncommitted work in the tree, and leaving a mutant behind is the
  highest-severity failure mode of this exercise.
* **Cheap-first, then full.** A mutant is declared KILLED as soon as any lane
  fails, so the lane most likely to catch it runs first (`focus`). A mutant is
  only declared SURVIVED after the *complete* suite has run against it —
  a kill from a cheap lane is sound, a survival from one is not.
* **Timeouts are not kills.** A hung suite is recorded as TIMEOUT and excluded
  from the mutation score numerator, because a timeout is not an assertion.

Everything runs inside the toolchain container: the host has no CUDA toolkit.

Usage:
  tools/mutation/run.py --list
  tools/mutation/run.py --catalogue tools/mutation/catalogue.json [--only M01,M02]
  tools/mutation/run.py --baseline-only    # verify the clean tree is green first
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV_IMG = os.environ.get('IMP_DEV_IMG', 'imp:toolchain')
DEV_DIR = os.environ.get('IMP_DEV_DIR', 'build-dev')
MODELS = os.environ.get('IMP_MODELS_DIR',
                        os.path.join(os.path.expanduser('~'), 'models'))
BACKUP = Path(os.environ.get('IMP_MUT_BACKUP',
                             os.path.join(tempfile.gettempdir(), 'imp-mutation-backup')))

# Model env for the GPU lanes — without these ~63 tests skip silently and the
# run looks green for the wrong reason (see loop/run_gpu_suite.sh).
MODEL_ENV = {
    'IMP_TEST_MODEL': '/models/Qwen3-8B-Q8_0.gguf',
    'IMP_TEST_GGUF': '/models/Qwen3-4B-Instruct-2507-Q8_0.gguf',
    'IMP_TEST_MODEL_QWEN4B': '/models/Qwen3-4B-Instruct-2507-Q8_0.gguf',
    'IMP_TEST_MODEL_LLAMA': '/models/Llama-3.2-3B-Instruct-Q8_0.gguf',
    'IMP_TEST_MODEL_GDN': '/models/Qwen3.5-4B-mxfp4.gguf',
    'IMP_TEST_MODEL_GEMMA4': '/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf',
    'IMP_TEST_MOE_MODEL': '/models/gpt-oss-20b-mxfp4.gguf',
    'IMP_TEST_MODEL_DEEPSEEK': '/models/DeepSeek-V2-Lite',
    'IMP_TEST_MODEL_QWEN3VL': '/models/Qwen3-VL-4B-GGUF/Qwen3-VL-4B-Instruct-Q4_K_M.gguf',
    'IMP_TEST_MMPROJ': '/models/Qwen3-VL-4B-GGUF/mmproj-F16.gguf',
    'IMP_TEST_MMPROJ_GEMMA4': '/models/gemma-3-4b-vl/mmproj-gemma4-26b-bf16.gguf',
    'IMP_VISION_TEST_IMAGE': '/src/tests/fixtures/vision_test_64.png',
    'IMP_TEST_IMAGE_ALT': '/src/tests/fixtures/vision_test_green_bar.png',
}

# Test binaries, cheapest first. `focus` in a mutant entry promotes one of these.
BINARIES = ['test-core', 'test-text', 'test-kv', 'test-quant', 'test-moe-gdn',
            'test-compute', 'test-attention', 'test-e2e']

# What GitHub CI actually executes on a PR: `ctest -L unit` — test-core,
# test-text and the CPU-only slice of test-e2e, on a runner with no GPU.
# Recorded separately from the local full-suite verdict, because "a mutant the
# merge gate cannot see" and "a mutant nothing in the repo can see" are two
# different findings and only the second is a missing test.
CI_UNIT_E2E_FILTER = ('BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:'
                      'EndToEndTest.*:StubModelTest.LoadStubModel:'
                      'StubModelTest.TokenizeStub')
CI_LANES = [('test-core', None), ('test-text', None),
            ('test-e2e', ['--gtest_filter=' + CI_UNIT_E2E_FILTER])]


def docker(cmd, workdir='/src', gpus=False, timeout=None, env=None):
    argv = ['docker', 'run', '--rm']
    if gpus:
        argv += ['--gpus', 'all', '-v', f'{MODELS}:/models']
    argv += ['-v', f'{REPO}:/src', '-w', workdir]
    for k, v in (env or {}).items():
        argv += ['-e', f'{k}={v}']
    argv += [DEV_IMG] + cmd
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b'')
        err = (e.stderr or b'')
        if isinstance(out, bytes):
            out = out.decode(errors='replace')
        if isinstance(err, bytes):
            err = err.decode(errors='replace')
        return 124, out + err


def build(timeout=1800):
    """Incremental ninja build inside the toolchain container."""
    rc, out = docker(['ninja', '-C', f'/src/{DEV_DIR}'], timeout=timeout)
    return rc, out


def run_binary(name, timeout=1800, extra_args=None):
    rc, out = docker(['./' + name] + (extra_args or []),
                     workdir=f'/src/{DEV_DIR}', gpus=True,
                     timeout=timeout, env=MODEL_ENV)
    return rc, out


def crashed(rc, output):
    """True when the binary died instead of reporting failures.

    A SIGFPE/SIGSEGV leaves no `[  FAILED  ]` line, so a kill-by-crash reads as
    "no failures" and the mutant is scored SURVIVED. That happened once (M46,
    rc=136 = SIGFPE from `pos_ % 0` after removing the zero-alignment guard) and
    is exactly the shape of a harness that flatters the suite. A run that neither
    exits 0 nor names a failing test did not pass.
    """
    if rc in (0, 124):
        return False
    return not failing_tests(output)


def failing_tests(output):
    """gtest names reported as FAILED, in order."""
    names = []
    for line in output.splitlines():
        s = line.strip()
        if s.startswith('[  FAILED  ]') and '.' in s:
            n = s[len('[  FAILED  ]'):].strip().split(' ')[0]
            if n and n not in names and not n[0].isdigit():
                names.append(n)
    return names


def apply_mutant(m):
    """Write the mutated file. Returns (path, original_bytes)."""
    path = REPO / m['file']
    original = path.read_bytes()
    text = original.decode()
    find, repl = m['find'], m['replace']
    n = text.count(find)
    if n != 1:
        raise RuntimeError(
            f"{m['id']}: anchor matches {n} times in {m['file']} (need exactly 1)")
    path.write_bytes(text.replace(find, repl, 1).encode())
    return path, original


def restore(path, original):
    path.write_bytes(original)


def tree_clean(allow_prefixes=('tests/', 'loop/', 'docs/audit/', 'tools/mutation/')):
    rc = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO,
                        capture_output=True, text=True)
    dirty = []
    for line in rc.stdout.splitlines():
        f = line[3:].strip()
        if not any(f.startswith(p) for p in allow_prefixes):
            dirty.append(line)
    return (not dirty), dirty


def save_patch(m):
    rc = subprocess.run(['git', 'diff', '--', m['file']], cwd=REPO,
                        capture_output=True, text=True)
    out = REPO / 'loop' / 'mutants' / f"{m['id']}.patch"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# {m['id']} [{m['category']}] {m['file']}\n"
              f"# {m['desc']}\n")
    out.write_text(header + rc.stdout)


def run_one(m, log_dir, full_timeout):
    """Returns a result dict. Guarantees the file is restored."""
    res = {'id': m['id'], 'category': m['category'], 'file': m['file'],
           'desc': m['desc'], 'status': 'ERROR', 'killed_by': None,
           'lanes': [], 'seconds': 0.0}
    t0 = time.time()
    path = original = None
    try:
        path, original = apply_mutant(m)
        save_patch(m)

        rc, out = build()
        (log_dir / f"{m['id']}-build.log").write_text(out[-40000:])
        if rc != 0:
            # A mutant that does not compile is not a test-suite result; it is a
            # broken mutant. Report it as such rather than counting it killed.
            res['status'] = 'BUILD_FAIL'
            return res

        # --- CI lane first (always run in full; ~0.4 s) ------------------
        # This answers "would the merge gate have caught it", which is a
        # separate question from "does the repo contain a test that catches
        # it". Never short-circuited, because both answers are wanted.
        res['ci_killed_by'] = None
        for b, extra in CI_LANES:
            rc, out = run_binary(b, timeout=full_timeout, extra_args=extra)
            tag = b + ('-ci' if extra else '')
            (log_dir / f"{m['id']}-{tag}.log").write_text(out[-200000:])
            fails = failing_tests(out)
            crash = crashed(rc, out)
            res['lanes'].append({'binary': tag, 'rc': rc, 'failed': fails,
                                 'crashed': crash, 'ci': True})
            if (fails or crash) and res['ci_killed_by'] is None:
                res['ci_killed_by'] = (f"{tag}: " + ', '.join(fails[:5])) if fails \
                    else f"{tag}: crashed (rc={rc})"

        if res['ci_killed_by']:
            res['status'] = 'KILLED'
            res['killed_by'] = res['ci_killed_by']
            return res

        done = {b for b, _ in CI_LANES if _ is None}
        order = ([m['focus']] if m.get('focus') else []) + \
                [b for b in BINARIES if b != m.get('focus')]
        for b in order:
            if b in done:
                continue
            done.add(b)
            rc, out = run_binary(b, timeout=full_timeout)
            (log_dir / f"{m['id']}-{b}.log").write_text(out[-200000:])
            fails = failing_tests(out)
            crash = crashed(rc, out)
            res['lanes'].append({'binary': b, 'rc': rc, 'failed': fails, 'crashed': crash})
            if rc == 124:
                res['status'] = 'TIMEOUT'
                return res
            if fails or crash:
                res['status'] = 'KILLED'
                res['killed_by'] = (f"{b}: " + ', '.join(fails[:5])) if fails \
                    else f"{b}: crashed (rc={rc})"
                return res
        res['status'] = 'SURVIVED'
        return res
    except Exception as e:  # noqa: BLE001 — must still restore below
        res['status'] = 'ERROR'
        res['error'] = str(e)
        return res
    finally:
        if path is not None and original is not None:
            restore(path, original)
        res['seconds'] = round(time.time() - t0, 1)
        ok, dirty = tree_clean()
        res['tree_clean_after'] = ok
        if not ok:
            res['dirty'] = dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalogue', default=str(Path(__file__).parent / 'catalogue.json'))
    ap.add_argument('--only', default='')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--baseline-only', action='store_true')
    ap.add_argument('--timeout', type=int, default=2400)
    ap.add_argument('--out', default=str(REPO / 'loop' / 'evidence' / 'mutation-results.json'))
    args = ap.parse_args()

    catalogue = json.loads(Path(args.catalogue).read_text())['mutants']
    if args.only:
        want = set(args.only.split(','))
        catalogue = [m for m in catalogue if m['id'] in want]

    if args.list:
        for m in catalogue:
            print(f"{m['id']:5s} {m['category']:14s} {m['file']}: {m['desc']}")
        return 0

    ok, dirty = tree_clean()
    if not ok:
        print('REFUSING TO START: production tree is dirty:', file=sys.stderr)
        print('\n'.join(dirty), file=sys.stderr)
        return 2

    log_dir = REPO / 'loop' / 'evidence' / 'mutants'
    log_dir.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)

    print('[baseline] building clean tree ...', flush=True)
    rc, out = build()
    if rc != 0:
        print('baseline build FAILED', file=sys.stderr)
        (log_dir / 'baseline-build.log').write_text(out[-40000:])
        return 2
    baseline = {}
    lanes = [(b, None) for b in BINARIES] + \
            [(b, e) for b, e in CI_LANES if e is not None]
    for b, extra in lanes:
        rc, out = run_binary(b, timeout=args.timeout, extra_args=extra)
        tag = b + ('-ci' if extra else '')
        (log_dir / f'baseline-{tag}.log').write_text(out[-200000:])
        baseline[tag] = {'rc': rc, 'failed': failing_tests(out),
                         'crashed': crashed(rc, out)}
        print(f'  baseline {tag}: rc={rc} failed={baseline[tag]["failed"]}', flush=True)
    (REPO / 'loop' / 'evidence' / 'mutation-baseline.json').write_text(
        json.dumps(baseline, indent=1))
    if args.baseline_only:
        return 0

    results = []
    for i, m in enumerate(catalogue, 1):
        print(f"[{i}/{len(catalogue)}] {m['id']} {m['category']}: {m['desc']}",
              flush=True)
        r = run_one(m, log_dir, args.timeout)
        # Discount tests that were already red on the clean tree.
        if r['status'] == 'KILLED':
            for lane in r['lanes']:
                if lane.get('crashed'):
                    continue  # a crash is never "pre-existing" — the baseline ran clean
                pre = set(baseline.get(lane['binary'], {}).get('failed', []))
                new = [f for f in lane['failed'] if f not in pre]
                if lane['failed'] and not new:
                    r['status'] = 'SURVIVED'
                    r['killed_by'] = None
                    r['note'] = 'only pre-existing failures — not a kill'
        print(f"    -> {r['status']} ({r['seconds']}s) {r['killed_by'] or ''}",
              flush=True)
        if not r.get('tree_clean_after', True):
            print('    !! TREE DIRTY AFTER REVERT — STOPPING', file=sys.stderr)
            results.append(r)
            break
        results.append(r)
        Path(args.out).write_text(json.dumps(results, indent=1))

    Path(args.out).write_text(json.dumps(results, indent=1))
    killed = sum(1 for r in results if r['status'] == 'KILLED')
    scored = sum(1 for r in results if r['status'] in ('KILLED', 'SURVIVED'))
    print(f'\nMutation score: {killed}/{scored} = '
          f'{100.0 * killed / scored if scored else 0:.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
