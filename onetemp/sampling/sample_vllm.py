#!/usr/bin/env python3
"""Sample n completions per (question, setting) with vLLM, as the paper's runs did (Appendix C and I). One GPU.

    # MATH-500, Mistral 7B Instruct: 16 temperatures x 1,500 samples on all 500 questions
    python -m onetemp.sampling.sample_vllm --run math500/mistral-7b --out runs/math500_mistral-7b

    # Hi-ToM, Qwen3-0.6B: first stage (500 samples at 31 temperatures, all 1,200 questions), then the screen,
    # then the second stage (+1,000) on the kept questions
    python -m onetemp.sampling.sample_vllm --run hitom/qwen3-0.6b --stage screen --out runs/hitom_q06_screen
    python -m onetemp.sampling.screen --rule hitom --counts counts_screen.jsonl --out kept.json
    python -m onetemp.sampling.sample_vllm --run hitom/qwen3-0.6b --stage topup --ids-file kept.json --out runs/hitom_q06_topup

    # min_p at a fixed temperature (Appendix I): the setting axis is min_p, temperature is tau
    python -m onetemp.sampling.sample_vllm --run minp/qwen3-0.6b/tau2.0 --out runs/minp_q06_tau2

Several GPUs: launch.py starts one process of this script per GPU (questions or samples split between them).

WHAT EACH STEP DOES
 1. The run profile (paper_runs.py) gives the settings grid, the stage's samples per cell and questions, max_tokens,
    the vLLM engine arguments and the extra sampling arguments of the paper's run. Any of them can be overridden;
    the resolved values are written to <out>/RUN.json and into every output line.
 2. Prompts: --prompts <file from build_prompts.py>, or built here the same way. They are token ids, sent unchanged.
 3. Guard: vLLM silently shortens max_tokens when prompt + max_tokens exceeds max_model_len, so the script refuses
    to start in that case (unless --allow-clamp; the Qwen3-8B LiveCodeBench run of the paper had 5 such prompts).
 4. Cells are visited question by question (all settings of a question before the next question). batching=cell
    sends one request per (question, setting) with SamplingParams(n = all samples of the cell); batching=question
    sends all settings of a question in one generate call. Either way each sample is an independent draw from the
    model at that setting; the pattern changes only how the random draws interleave.
 5. Each finished cell is written atomically to its own file, so a stopped run resumes where it stopped (a cell is
    skipped if its file exists). One directory holds one sampling configuration: a resumed or parallel worker whose
    configuration differs from <out>/RUN.json is refused.

OUTPUT, read by the graders: <out>/cells/<setting>/<question>.jsonl.gz, one JSON line per cell
    {"benchmark", "model" (HF id), "id", "temperature", "min_p" (null on temperature sweeps), "n",
     "completions": [text, ...], "finish_reasons": ["stop"|"length", ...], "stop_reasons", "num_output_tokens",
     "stage", "run", "part" ([worker, workers] when the samples of a cell are split over GPUs, else null),
     ["lcb_version", "lcb_row" on LiveCodeBench], "provenance": {sampling, engine, seed, versions, GPU, prompt hashes}}
With --split samples a cell is several files <question>.part<w>of<W>.jsonl.gz whose counts the graders add up.
The exit code is 0 only when every cell of this worker is on disk; a stopped worker (SIGTERM) exits 3.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
from . import build_prompts as bp   # noqa: E402
from . import paper_runs as pr      # noqa: E402

STOP = {"flag": False}
IDENTITY_KEYS = ("run", "stage", "benchmark", "model", "axis", "tau", "settings", "n_total", "max_tokens",
                 "sampling_extra", "text_mode", "split", "num_workers", "prompt_style", "prompt_set_sha256",
                 "engine")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def fnum(x):
    """Compact, exact-enough float text for file names: 0.075 -> '0.075', 2.0 -> '2'."""
    return f"{float(x):g}"


def setting_dir(axis, setting, tau):
    return f"t{fnum(setting)}" if axis == "temperature" else f"t{fnum(tau)}_mp{fnum(setting)}"


def slug(qid):
    return re.sub(r"[^A-Za-z0-9]+", "_", qid).strip("_")


def cell_path(out, axis, setting, tau, qid, part):
    name = slug(qid) + (f".part{part[0]}of{part[1]}" if part else "") + ".jsonl.gz"
    return os.path.join(out, "cells", setting_dir(axis, setting, tau), name)


def write_gz_atomic(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp_{os.getpid()}_{os.path.basename(path)}")
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            gz.write(data)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(tmp, path)


def write_json_atomic(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".tmp_{os.getpid()}_{os.path.basename(path)}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def parse_settings(s):
    vals = [float(x) for x in s.split(",") if x.strip()]
    if not vals or len(set(vals)) != len(vals):
        raise SystemExit(f"--settings must be distinct numbers, got {s!r}")
    return vals


def parse_engine_args(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--engine-arg takes KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def samples_for_worker(n_total, worker, workers):
    """Split n samples over workers as evenly as possible (the first n % workers get one more)."""
    return n_total // workers + (1 if worker < n_total % workers else 0)


def resolve_config(a):
    run = pr.get_run(a.run)
    stage = pr.stage_of(run, a.stage or run["stages"][0]["name"])
    mk = run["model"]
    model_repo = a.model or pr.MODELS[mk]["repo"]
    revision = a.revision or (pr.MODELS[mk]["revision"] if model_repo == pr.MODELS[mk]["repo"] else None)
    axis = run["axis"]
    tau = a.tau if a.tau is not None else run.get("tau")
    settings = parse_settings(a.settings) if a.settings else list(stage.get("settings") or run["settings"])
    engine = dict(run["engine"])
    engine.update(parse_engine_args(a.engine_arg))
    split = a.split or run["split"]
    workers = a.num_workers
    if not (0 <= a.worker < workers):
        raise SystemExit(f"--worker must be in [0, {workers})")
    cfg = {"run": a.run, "stage": stage["name"], "benchmark": run["benchmark"], "model": model_repo,
           "revision": revision, "axis": axis, "tau": tau, "settings": settings,
           "n_total": int(a.n if a.n is not None else stage["n"]), "max_tokens": int(a.max_tokens or run["max_tokens"]),
           "sampling_extra": dict(run["sampling"]), "text_mode": run["text"],
           "batching": a.batching or run["batching"], "split": split, "num_workers": workers, "worker": a.worker,
           "engine": engine, "allow_clamp": bool(a.allow_clamp or run.get("allow_clamp", False)),
           "prompt_style": run["prompt_style"], "paper_vllm": run["vllm"]}
    if axis == "min_p" and tau is None:
        raise SystemExit("a min_p run needs a fixed temperature (--tau)")
    if axis == "temperature" and any(t < 0 for t in settings):
        raise SystemExit("temperatures must be >= 0")
    if axis == "min_p" and any(not (0.0 <= m < 1.0) for m in settings):
        raise SystemExit("min_p values must be in [0, 1)")
    # Engine seed: the profile's policy for a run's first stage, a fresh random seed for every later stage, unless
    # --seed. A fresh engine given the same first request with the same seed can replay the same draws, so a second
    # stage must not reuse the first stage's seed (its first question may be the first stage's first question).
    # For the same reason, workers that split the SAMPLES of a cell get distinct seeds (base + worker) unless
    # --same-seed. Every seed is recorded in every output line.
    if a.seed != "default":
        seed = a.seed
    elif stage["name"] == run["stages"][0]["name"]:
        seed = run["engine_seed"]
    else:
        seed = "random"
    if seed == "random":
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    elif seed in (None, "none"):
        seed = None
    else:
        seed = int(seed)
    if split == "samples" and workers > 1 and not a.same_seed:
        seed = (seed or 0) + a.worker
    cfg["seed"] = seed
    return run, cfg


def select_questions(a, run, cfg, prompts):
    ids, src = bp.resolve_question_ids(run, cfg["stage"], a.ids_file)
    by_id = {p["id"]: p for p in prompts}
    if ids is None:
        chosen = list(prompts)
    else:
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} requested ids have no prompt (e.g. {missing[:3]}); build prompts for them")
        want = set(ids)
        chosen = [p for p in prompts if p["id"] in want]
    for p in chosen:
        if p.get("benchmark") != cfg["benchmark"] or p.get("model") != cfg["model"]:
            raise SystemExit(f"prompt {p['id']} is for {p.get('benchmark')}/{p.get('model')}, the run is "
                             f"{cfg['benchmark']}/{cfg['model']}")
        if p.get("prompt_style") != cfg["prompt_style"] and not a.any_style:
            raise SystemExit(f"prompt {p['id']} has style {p.get('prompt_style')}, the run uses "
                             f"{cfg['prompt_style']} (pass --any-style to allow)")
    slugs = [slug(p["id"]) for p in chosen]
    if len(set(slugs)) != len(slugs):
        raise SystemExit("two question ids map to the same file name")
    h = hashlib.sha256()
    for p in chosen:
        h.update((p["id"] + "\0" + p["prompt_ids_sha256"] + "\n").encode())
    cfg["prompt_set_sha256"] = h.hexdigest()
    cfg["questions_source"] = src
    cfg["n_questions_all_workers"] = len(chosen)
    if cfg["split"] == "questions":
        mine = chosen[cfg["worker"]::cfg["num_workers"]]          # round robin, as the paper's launchers dealt them
        n_here = cfg["n_total"]
        part = None
    elif cfg["split"] == "samples":
        mine = chosen
        n_here = samples_for_worker(cfg["n_total"], cfg["worker"], cfg["num_workers"])
        part = [cfg["worker"], cfg["num_workers"]] if cfg["num_workers"] > 1 else None
    else:
        raise SystemExit(f"--split must be questions or samples, got {cfg['split']!r}")
    return mine, n_here, part


def check_lengths(qs, cfg):
    mml = cfg["engine"].get("max_model_len")
    if mml is None:
        return []
    too_long = [(p["id"], p["n_prompt_tokens"]) for p in qs if p["n_prompt_tokens"] >= mml]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts do not fit max_model_len={mml}: {too_long[:5]}")
    clamped = [(p["id"], p["n_prompt_tokens"]) for p in qs if p["n_prompt_tokens"] + cfg["max_tokens"] > mml]
    if clamped and not cfg["allow_clamp"]:
        raise SystemExit(f"{len(clamped)} prompts + max_tokens={cfg['max_tokens']} exceed max_model_len={mml}; vLLM "
                         f"would silently give them fewer new tokens: {clamped[:5]}. Raise max_model_len "
                         f"(--engine-arg max_model_len=N) or pass --allow-clamp to run them as the paper did.")
    return clamped


def run_identity(cfg):
    ident = {k: cfg[k] for k in IDENTITY_KEYS}
    if cfg["split"] == "questions":
        ident["num_workers"] = None        # file names do not depend on the worker: any number may resume
    return ident


def check_run_identity(out, cfg, force):
    """One directory = one sampling configuration. The first worker writes RUN.json (exclusive create); every
    other worker, and every resume, must match it."""
    ident = run_identity(cfg)
    path = os.path.join(out, "RUN.json")
    os.makedirs(out, exist_ok=True)
    body = json.dumps({"identity": ident, "created_utc": utc(), "paper_vllm": cfg["paper_vllm"]}, indent=1)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body + "\n")
        return
    except FileExistsError:
        pass
    for _ in range(50):                       # another worker may be writing it right now
        try:
            with open(path, encoding="utf-8") as fh:
                have = json.load(fh)["identity"]
            break
        except (ValueError, KeyError):
            time.sleep(0.1)
    else:
        raise SystemExit(f"{path} is unreadable")
    diff = {k: (have.get(k), ident[k]) for k in IDENTITY_KEYS if have.get(k) != ident[k]}
    if diff and not force:
        raise SystemExit(f"{out} holds a different sampling configuration (use one directory per configuration); "
                         f"differences (existing, now): {json.dumps(diff)[:2000]}")


def planned_cells(qs, cfg, part, out):
    cells = []
    for p in qs:
        for s in cfg["settings"]:
            cells.append((p, s, cell_path(out, cfg["axis"], s, cfg["tau"], p["id"], part)))
    return cells


def make_engine(cfg, a):
    """(llm, SamplingParams class, decode tokenizer or None, version info)."""
    if a.engine == "fake":
        from . import fake_engine as fe
        llm = fe.FakeLLM(cfg["benchmark"], seed=cfg["seed"], **cfg["engine"])
        tok = fe.FakeTokenizer() if cfg["text_mode"] == "decode_ids" else None
        return llm, fe.FakeSamplingParams, tok, {"engine_impl": "fake (test double, not vLLM)"}
    import vllm
    from vllm import LLM, SamplingParams
    kw = dict(cfg["engine"])
    if a.model_path:
        kw["model"] = a.model_path
    else:
        kw["model"] = cfg["model"]
        if cfg["revision"]:
            kw["revision"] = cfg["revision"]
            kw["tokenizer_revision"] = cfg["revision"]
    if cfg["seed"] is not None:
        kw["seed"] = cfg["seed"]
    llm = LLM(disable_log_stats=True, **kw)
    tok = None
    if cfg["text_mode"] == "decode_ids":       # the runs decoded the token ids with the model's HF tokenizer
        from transformers import AutoTokenizer
        src = a.model_path or cfg["model"]
        tok = AutoTokenizer.from_pretrained(src, revision=None if a.model_path else cfg["revision"])
    info = {"engine_impl": "vllm", "vllm": vllm.__version__}
    try:
        import torch
        import transformers
        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as ex:                    # provenance only
        info["version_error"] = repr(ex)
    try:
        info["engine_seed_live"] = llm.llm_engine.model_config.seed
    except Exception:
        pass
    return llm, SamplingParams, tok, info


def sampling_kwargs(cfg, setting, n):
    kw = dict(n=n, max_tokens=cfg["max_tokens"])
    if cfg["axis"] == "temperature":
        kw["temperature"] = float(setting)
    else:
        kw["temperature"] = float(cfg["tau"])
        if setting > 0:                  # min_p = 0 is plain sampling (vLLM's default), not passed, as in the runs
            kw["min_p"] = float(setting)
    kw.update(cfg["sampling_extra"])
    return kw


def cell_record(cfg, p, setting, part, outs, texts, sp, info, meta):
    min_p = None if cfg["axis"] == "temperature" else float(setting)
    temp = float(setting) if cfg["axis"] == "temperature" else float(cfg["tau"])
    rec = {"benchmark": cfg["benchmark"], "model": cfg["model"], "id": p["id"], "temperature": temp, "min_p": min_p,
           "n": len(texts), "completions": texts, "finish_reasons": [o.finish_reason for o in outs],
           "stop_reasons": [o.stop_reason for o in outs], "num_output_tokens": [len(o.token_ids) for o in outs],
           "stage": cfg["stage"], "run": cfg["run"], "part": part}
    for k in ("lcb_version", "lcb_row"):
        if k in p:
            rec[k] = p[k]
    rec["provenance"] = dict(meta, sampling_params_repr=repr(sp), prompt_ids_sha256=p["prompt_ids_sha256"],
                             prompt_text_sha256=p["prompt_text_sha256"], n_prompt_tokens=p["n_prompt_tokens"],
                             prompt_style=p["prompt_style"], **info)
    return rec


def on_term(signum, frame):
    STOP["flag"] = True
    print(f"[{utc()}] signal {signum}: stopping after the current request", file=sys.stderr, flush=True)


def build_parser(description=__doc__):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    bp.add_common_args(ap)
    ap.add_argument("--prompts", help="prompts JSONL from build_prompts.py (default: build them now)")
    ap.add_argument("--any-style", action="store_true", help="accept prompts of another style than the run's")
    ap.add_argument("--out", required=True, help="output directory, one per sampling configuration")
    g = ap.add_argument_group("overrides (default: the run profile)")
    g.add_argument("--settings", help="comma-separated temperatures (or min_p values on a min_p run)")
    g.add_argument("--tau", type=float, help="the fixed temperature of a min_p run")
    g.add_argument("--n", type=int, help="samples per (question, setting) in this stage, over all workers")
    g.add_argument("--max-tokens", type=int, help="token limit per sample")
    g.add_argument("--engine-arg", action="append", metavar="KEY=VALUE",
                   help="override a vLLM LLM() argument (JSON value), e.g. max_num_seqs=256; repeatable")
    g.add_argument("--model-path", help="load the weights from this local directory (the records keep --model)")
    g.add_argument("--seed", default="default",
                   help="engine seed: an integer, 'random', 'none' (vLLM's default) or 'default' (the profile's "
                        "policy on a run's first stage, a fresh random seed on later stages)")
    g.add_argument("--same-seed", action="store_true",
                   help="with --split samples, give every worker the same engine seed (as the paper's LiveCodeBench "
                        "runs did) instead of seed + worker")
    g.add_argument("--batching", choices=["cell", "question"], help="one request per cell, or per question")
    g.add_argument("--split", choices=["questions", "samples"], help="what --num-workers split: the questions, "
                                                                     "or the samples of every cell")
    g.add_argument("--worker", type=int, default=0, help="this worker's index (launch.py sets it)")
    g.add_argument("--num-workers", type=int, default=1, help="number of workers sharing --out")
    g.add_argument("--allow-clamp", action="store_true",
                   help="run prompts for which prompt + max_tokens > max_model_len (vLLM shortens them)")
    ap.add_argument("--engine", choices=["vllm", "fake"], default="vllm",
                    help="'fake' runs the whole pipeline with a deterministic stand-in (no GPU), for tests")
    ap.add_argument("--max-cells", type=int, help="stop after this many new cells (for short tests)")
    ap.add_argument("--plan", action="store_true", help="print the resolved configuration and the cells, then exit")
    ap.add_argument("--force", action="store_true", help="accept an --out whose RUN.json differs")
    return ap


def prepare(a, prompts=None, prompts_sha=None):
    """Resolve the configuration and this worker's cells (no engine). Shared with launch.py."""
    if not a.run:
        raise SystemExit("--run is required (see paper_runs.py for the profiles)")
    run, cfg = resolve_config(a)
    if prompts is None:
        if a.prompts:
            prompts = bp.read_prompts(a.prompts)
            prompts_sha = file_sha256(a.prompts)
        else:
            a.stage = cfg["stage"]
            prompts, _ = bp.make_prompts(a)
    qs, n_here, part = select_questions(a, run, cfg, prompts)
    clamped = check_lengths(qs, cfg)
    cells = planned_cells(qs, cfg, part, a.out)
    return {"run": run, "cfg": cfg, "qs": qs, "n_here": n_here, "part": part, "clamped": clamped,
            "cells": cells, "prompts_sha": prompts_sha}


def main():
    a = build_parser().parse_args()
    prep = prepare(a)
    run, cfg, qs, n_here, part = prep["run"], prep["cfg"], prep["qs"], prep["n_here"], prep["part"]
    clamped, cells, prompts_sha = prep["clamped"], prep["cells"], prep["prompts_sha"]
    todo = [c for c in cells if not os.path.exists(c[2])]
    if a.plan:
        show = {k: v for k, v in cfg.items()}
        show.update(questions_this_worker=len(qs), samples_per_cell_this_worker=n_here, part=part,
                    cells=len(cells), cells_todo=len(todo), clamped_prompts=len(clamped))
        print(json.dumps(show, indent=1))
        return 0
    if n_here <= 0:
        raise SystemExit("this worker has no samples to draw")
    check_run_identity(a.out, cfg, a.force)
    wpath = os.path.join(a.out, "workers", f"w{cfg['worker']}.json")
    status = {"worker": cfg["worker"], "num_workers": cfg["num_workers"], "pid": os.getpid(), "start_utc": utc(),
              "cells": len(cells), "cells_todo_at_start": len(todo), "state": "loading", "seed": cfg["seed"]}
    write_json_atomic(wpath, status)
    if clamped:
        print(f"WARNING: {len(clamped)} prompts get fewer than max_tokens new tokens (allowed): {clamped[:5]}",
              file=sys.stderr)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    llm, SP, dec_tok, info = make_engine(cfg, a)
    sampler_sha = file_sha256(os.path.abspath(__file__))
    meta = {"run": cfg["run"], "stage": cfg["stage"], "model_revision": cfg["revision"], "engine": cfg["engine"],
            "engine_seed": cfg["seed"], "split": {"mode": cfg["split"], "worker": cfg["worker"],
                                                 "num_workers": cfg["num_workers"]},
            "batching": cfg["batching"], "text_mode": cfg["text_mode"], "prompts_sha256": prompts_sha,
            "prompt_set_sha256": cfg["prompt_set_sha256"], "sampler_sha256": sampler_sha}
    status["state"] = "sampling"
    write_json_atomic(wpath, status)
    print(f"[{utc()}] worker {cfg['worker']}/{cfg['num_workers']}: {len(qs)} questions x {len(cfg['settings'])} "
          f"settings, {n_here} samples per cell, {len(todo)} of {len(cells)} cells to do", file=sys.stderr, flush=True)

    request_index = new_cells = 0
    by_q = {}
    for c in todo:
        by_q.setdefault(c[0]["id"], []).append(c)
    t_run = time.time()
    for qid in [p["id"] for p in qs]:
        if STOP["flag"] or (a.max_cells is not None and new_cells >= a.max_cells):
            break
        group = by_q.get(qid, [])
        if a.max_cells is not None:
            group = group[:a.max_cells - new_cells]
        batches = [[c] for c in group] if cfg["batching"] == "cell" else ([group] if group else [])
        for batch in batches:
            if STOP["flag"]:
                break
            p = batch[0][0]
            sps = [SP(**sampling_kwargs(cfg, c[1], n_here)) for c in batch]
            t0, t0u = time.time(), utc()
            results = llm.generate([{"prompt_token_ids": p["prompt_token_ids"]}] * len(batch),
                                   sps if len(sps) > 1 else sps[0], use_tqdm=False)
            dt = time.time() - t0
            for c, sp, res in zip(batch, sps, results):
                outs = sorted(res.outputs, key=lambda o: o.index)
                if len(outs) != n_here or list(res.prompt_token_ids) != list(p["prompt_token_ids"]):
                    raise RuntimeError(f"{qid}: vLLM returned {len(outs)} outputs or a different prompt")
                if cfg["text_mode"] == "decode_ids":
                    texts = dec_tok.batch_decode([list(o.token_ids) for o in outs], skip_special_tokens=False)
                else:
                    texts = [o.text for o in outs]
                m = dict(meta, request_index=request_index, gen_seconds=round(dt, 3), utc_start=t0u, utc_end=utc(),
                         cells_in_request=len(batch))
                write_gz_atomic(c[2], cell_record(cfg, p, c[1], part, outs, texts, sp, info, m))
                new_cells += 1
            request_index += 1
        status.update(cells_written=new_cells, last_question=qid, elapsed_s=round(time.time() - t_run, 1))
        write_json_atomic(wpath, status)

    missing = [c[2] for c in cells if not os.path.exists(c[2])]
    status.update(end_utc=utc(), cells_written=new_cells, cells_missing=len(missing),
                  state="complete" if not missing else ("stopped" if STOP["flag"] else "incomplete"))
    write_json_atomic(wpath, status)
    print(f"[{utc()}] worker {cfg['worker']}: {new_cells} cells written, {len(missing)} of {len(cells)} missing "
          f"({status['state']})", file=sys.stderr, flush=True)
    return 0 if not missing else 3


if __name__ == "__main__":
    sys.exit(main())
