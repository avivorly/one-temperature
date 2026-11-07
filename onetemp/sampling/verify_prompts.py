#!/usr/bin/env python3
"""Check a prompt file against the paper's prompt hashes (expected_prompts/<benchmark>__<model>.json).

    python -m onetemp.sampling.build_prompts --run hitom/qwen3-0.6b --out p.jsonl
    python -m onetemp.sampling.verify_prompts --prompts p.jsonl --expected onetemp/sampling/expected_prompts/hitom__qwen3-0.6b.json

The expected files hold, per question id, what the paper's runs recorded about the prompt they sent:
    n_tokens     the number of prompt tokens
    ids_sha256   sha256 of json.dumps(prompt token ids)            (when the runs stored the ids or their hash)
    text_sha256  sha256 of the prompt text                          (see the file's "text" field for which text)
A prompt passes when every recorded field equals the rebuilt one. The report lists how many passed, and the first
mismatches. Exit code 0 only if every expected id is present and passes.

For GPT-OSS on LiveCodeBench the runs stamped the render day into the prompt; those files hash the text with the
date masked ("Current date: YYYY-MM-DD"), and this script masks the rebuilt text the same way.
"""
import argparse
import hashlib
import json
import re
import sys

DATE_RE = re.compile(r"Current date: \d{4}-\d{2}-\d{2}\n")


def mask_date(text):
    return DATE_RE.sub("Current date: YYYY-MM-DD\n", text)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def compare(prompts, expected):
    exp = expected["prompts"]
    masked = expected.get("text") == "decoded prompt, date line masked"
    got = {p["id"]: p for p in prompts}
    ok, bad, missing = 0, [], []
    for qid, e in exp.items():
        p = got.get(qid)
        if p is None:
            missing.append(qid)
            continue
        diffs = []
        if "n_tokens" in e and e["n_tokens"] != p["n_prompt_tokens"]:
            diffs.append(f"n_tokens {p['n_prompt_tokens']} != {e['n_tokens']}")
        if "ids_sha256" in e and e["ids_sha256"] != ids_sha(p["prompt_token_ids"]):
            diffs.append("ids_sha256")
        if "text_sha256" in e:
            t = mask_date(p["prompt_text"]) if masked else p["prompt_text"]
            if e["text_sha256"] != sha(t):
                diffs.append("text_sha256")
        if diffs:
            bad.append((qid, diffs))
        else:
            ok += 1
    extra = [q for q in got if q not in exp]
    return ok, bad, missing, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True, help="prompts JSONL from build_prompts.py")
    ap.add_argument("--expected", required=True, help="expected_prompts/<benchmark>__<model>.json")
    a = ap.parse_args()
    with open(a.prompts, encoding="utf-8") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    with open(a.expected, encoding="utf-8") as fh:
        expected = json.load(fh)
    ok, bad, missing, extra = compare(prompts, expected)
    n = len(expected["prompts"])
    print(json.dumps({"expected": n, "identical": ok, "different": len(bad), "missing": len(missing),
                      "not_in_expected": len(extra), "fields": expected.get("fields"),
                      "first_differences": bad[:5], "first_missing": missing[:5]}, indent=1))
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
