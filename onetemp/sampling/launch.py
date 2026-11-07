#!/usr/bin/env python3
"""Multi-GPU launcher: one sample_vllm.py process per GPU, sharing one output directory.

    # questions dealt round robin over 4 GPUs; every cell comes from one GPU (MATH-500 and Hi-ToM runs)
    python -m onetemp.sampling.launch --gpus 0,1,2,3 --run math500/qwen3-0.6b --out runs/math500_q06

    # every GPU samples every question with its share of n (the LiveCodeBench runs: 1,500 = 4 x 375)
    python -m onetemp.sampling.launch --gpus 0,1,2,3 --split samples --run livecodebench/mistral-7b --lcb-dir LCB \\
        --out runs/lcb_mistral_main

Every argument other than --gpus, --stagger and --poll is passed to each worker unchanged (see
`python -m onetemp.sampling.sample_vllm --help`). The launcher:
 1. builds the prompts once (unless --prompts is given) and writes them to <out>/prompts.jsonl, so every worker
    sends the same token ids;
 2. plans every worker's cells with the workers' own code;
 3. starts worker i on GPU gpus[i] (CUDA_VISIBLE_DEVICES), a few seconds apart, logging to <out>/logs/;
 4. on SIGTERM or Ctrl-C, passes SIGTERM to the workers and waits for them (SIGTERM only: a killed GPU process
    can leave the GPU's memory held);
 5. when no worker is alive, checks the result by the FILES, not by exit codes: every planned cell file must
    exist, parse, and hold the planned number of samples for its question and setting. It prints what is missing
    and exits 0 only when nothing is. A stopped worker exits cleanly too, so an exit code of 0 from a worker
    alone never means its work is done.
Re-running the same command resumes: existing cells are kept and only the missing ones are sampled.
"""
import gzip
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))   # the folder that holds the onetemp package
from . import build_prompts as bp   # noqa: E402
from . import sample_vllm as sv     # noqa: E402

CHILDREN = []


def parse_gpus(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(str(i) for i in range(int(lo), int(hi) + 1))
        elif part:
            out.append(part)
    if not out or len(set(out)) != len(out):
        raise SystemExit(f"--gpus must list distinct GPU indices, got {s!r}")
    return out


def split_argv(argv):
    """Separate the launcher's own options from the ones passed to every worker."""
    own, rest, i = {"gpus": None, "stagger": 3.0, "poll": 60.0}, [], 0
    while i < len(argv):
        k = argv[i]
        key = k[2:].split("=", 1)[0] if k.startswith("--") else None
        if key in own:
            if "=" in k:
                own[key] = k.split("=", 1)[1]
                i += 1
            else:
                own[key] = argv[i + 1]
                i += 2
            continue
        rest.append(k)
        i += 1
    if own["gpus"] is None:
        raise SystemExit("--gpus is required, e.g. --gpus 0,1,2,3 or --gpus 0-7")
    for bad in ("--worker", "--num-workers", "--plan"):
        if any(x == bad or x.startswith(bad + "=") for x in rest):
            raise SystemExit(f"{bad} is set by the launcher")
    return own, rest


def interpreter_flags():
    """Give the workers the launcher's own interpreter flags (-I, -B, -X pycache_prefix)."""
    f = []
    if sys.flags.isolated:
        f.append("-I")
    if sys.flags.dont_write_bytecode:
        f.append("-B")
    if getattr(sys, "pycache_prefix", None):
        f += ["-X", f"pycache_prefix={sys.pycache_prefix}"]
    return f


def check_cell(path, n_expected, qid, setting_key):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
    except (OSError, ValueError) as ex:
        return f"unreadable ({ex})"
    if rec.get("id") != qid:
        return f"id {rec.get('id')!r}, expected {qid!r}"
    if [rec.get("temperature"), rec.get("min_p")] != list(setting_key):
        return f"setting {[rec.get('temperature'), rec.get('min_p')]}, expected {list(setting_key)}"
    if rec.get("n") != n_expected or len(rec.get("completions") or []) != n_expected:
        return f"{len(rec.get('completions') or [])} samples, expected {n_expected}"
    return None


def on_term(signum, frame):
    print(f"launcher: signal {signum}, passing SIGTERM to {len(CHILDREN)} workers", file=sys.stderr, flush=True)
    for p in CHILDREN:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)


def main(argv):
    if not argv or "-h" in argv or "--help" in argv:
        print(__doc__)
        print("launcher options: --gpus LIST (e.g. 0,1,2,3 or 0-7), --stagger SECONDS (3), --poll SECONDS (60)\n")
        print("worker options (passed through):\n")
        sv.build_parser().print_help()
        return 0
    own, rest = split_argv(argv)
    gpus = parse_gpus(own["gpus"])
    W = len(gpus)
    parser = sv.build_parser()
    a0 = parser.parse_args(rest + ["--worker", "0", "--num-workers", str(W)])
    if a0.engine != "fake":
        try:
            import vllm  # noqa: F401  (fail early, before planning, if vLLM is missing)
        except ImportError:
            raise SystemExit("vLLM is not installed in this environment (pip install -e \".[sampling]\")")
    out = a0.out
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)

    # 1. prompts once
    if a0.prompts:
        prompts_path = a0.prompts
    else:
        prompts_path = os.path.join(out, "prompts.jsonl")
        if not os.path.exists(prompts_path):
            a0.stage = a0.stage or sv.pr.get_run(a0.run)["stages"][0]["name"]
            recs, meta = bp.make_prompts(a0)
            bp.write_prompts(prompts_path, recs, meta)
        rest = rest + ["--prompts", prompts_path]
    prompts = bp.read_prompts(prompts_path)
    prompts_sha = sv.file_sha256(prompts_path)

    # 2. plan every worker with the workers' code
    plans = []
    for w in range(W):
        aw = parser.parse_args(rest + ["--worker", str(w), "--num-workers", str(W)])
        prep = sv.prepare(aw, prompts=prompts, prompts_sha=prompts_sha)
        plans.append(prep)
    cfg = plans[0]["cfg"]
    total = sum(len(p["cells"]) for p in plans)
    todo = sum(1 for p in plans for c in p["cells"] if not os.path.exists(c[2]))
    print(f"launcher: {cfg['run']} stage {cfg['stage']}, {cfg['n_questions_all_workers']} questions x "
          f"{len(cfg['settings'])} settings, n={cfg['n_total']} per cell, split={cfg['split']} over {W} GPUs: "
          f"{total} cell files planned, {todo} to do", file=sys.stderr, flush=True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    if todo:
        # 3. one process per GPU
        for w, g in enumerate(gpus):
            if not any(not os.path.exists(c[2]) for c in plans[w]["cells"]):
                print(f"launcher: worker {w} has nothing to do", file=sys.stderr)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            env["PYTHONPATH"] = PKG_ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            log = open(os.path.join(out, "logs", f"worker{w}_gpu{g}.log"), "a", encoding="utf-8")
            cmd = [sys.executable] + interpreter_flags() + ["-m", "onetemp.sampling.sample_vllm"] + rest + \
                  ["--worker", str(w), "--num-workers", str(W)]
            CHILDREN.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
            print(f"launcher: worker {w} on GPU {g}, pid {CHILDREN[-1].pid}", file=sys.stderr, flush=True)
            if w < W - 1:
                time.sleep(float(own["stagger"]))      # vLLM engines started at the same instant can race
        # 4. wait; report progress from the files
        while any(p.poll() is None for p in CHILDREN):
            time.sleep(float(own["poll"]))
            done = sum(1 for p in plans for c in p["cells"] if os.path.exists(c[2]))
            alive = sum(1 for p in CHILDREN if p.poll() is None)
            print(f"launcher: {done}/{total} cell files, {alive} workers alive", file=sys.stderr, flush=True)
    codes = [p.returncode for p in CHILDREN]

    # 5. verify by the files
    problems = []
    for prep in plans:
        c0 = prep["cfg"]
        for p, s, path in prep["cells"]:
            key = (float(s), None) if c0["axis"] == "temperature" else (float(c0["tau"]), float(s))
            if not os.path.exists(path):
                problems.append((path, "missing"))
                continue
            err = check_cell(path, prep["n_here"], p["id"], key)
            if err:
                problems.append((path, err))
    summary = {"run": cfg["run"], "stage": cfg["stage"], "gpus": gpus, "worker_exit_codes": codes,
               "cells_planned": total, "cells_bad_or_missing": len(problems), "complete": not problems,
               "checked_utc": sv.utc()}
    sv.write_json_atomic(os.path.join(out, "LAUNCH_CHECK.json"), dict(summary, first_problems=problems[:50]))
    print(json.dumps(summary), flush=True)
    if problems:
        for path, err in problems[:10]:
            print(f"  {err}: {path}", file=sys.stderr)
        print(f"launcher: INCOMPLETE, {len(problems)} of {total} cell files missing or bad; rerun the same "
              f"command to resume", file=sys.stderr)
        return 3
    print(f"launcher: complete, all {total} cell files verified", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
