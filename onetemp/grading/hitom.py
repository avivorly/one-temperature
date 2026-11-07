#!/usr/bin/env python3
"""Hi-ToM grader of the paper: completions -> counts.jsonl.

    python -m onetemp.grading.hitom --inputs runs/qwen3_8b_hitom --out counts_hitom.jsonl --summary summary.json
    python -m onetemp.grading.hitom --selftest

WHAT IT COMPUTES (Appendix C, "Datasets"). Hi-ToM is multiple choice: the prompt lists all 15 options, so the
reference answer is inside every prompt. The answer is therefore read from the completion only, never from the
prompt: it is the FIRST VALID OPTION in the completion, that is, the first mention of one of the question's own
options, by its name or by its letter given as an answer label. A completion that names no valid option fails.
Per (question, setting) it writes n = samples and c = correct samples (instance-weighted), the estimate c/n of
P_q(tau) in Section 2.

The reference answers and option lists come from the public Hi-ToM dataset (github.com/ying-hui-he/Hi-ToM_dataset,
Hi-ToM_data/Hi-ToM_data.json), downloaded at run time at a pinned commit and checked by sha256. A question's id is
"s{sample_id}_o{question_order}_{prompting_type}", for example "s12_o3_CoTP".

THE MATCHER (copied unchanged from the grader of the paper's runs; its contract hash, grader_hash(), is bf137bfc):
  * extract(completion, choices) and analyse(completion, choices) take the completion and the question's own
    option list ONLY. There is no record and no reference-answer argument, so the answer key cannot be consulted
    while the answer is extracted. grade(extracted, gold) rejects a missing answer before any comparison.
  * Matching is exact normalized equality (case-folded, underscores as spaces, whitespace collapsed) against THIS
    question's options: dict lookups only, no substring, prefix, fuzzy or regex match. `teal_drawer` cannot match
    `teal_drawer_2`, and a bare colour matches nothing.
  * A letter counts only as an answer label (a leading "K", "K.", "(K)", "**K**", or after "Answer:", "The answer
    is", "option" ...), resolved through this question's own option list.
  * THE SELECTION RULE, fixed before any results existed: the FIRST mention that is a valid option decides (the
    template asks for the answer first). A container-shaped token that is not on offer is skipped; a completion
    with no valid option fails as `not_a_choice` (it named something else) or `no_answer` (it named nothing).
    "Pass if any mention matches" would be in-sample oracle selection and is not available. The last valid
    mention is recorded and never used.
  * Buckets: correct, wrong_valid_choice, not_a_choice, no_answer, empty. They partition the samples.

THE ANSWER ADAPTER for chat-template outputs (answer_text() and its helpers copied unchanged; verdict() is the
production call without its bookkeeping fields). It is applied before the matcher:
  * Mistral 7B Instruct (vLLM's output text): the whole completion goes to the matcher.
  * Qwen3 models, thinking off: every literal <|im_end|> and <|endoftext|> is removed, then the whole text goes to
    the matcher.
  * GPT-OSS 20B, Harmony format with the final channel primed in the prompt: the text up to the first <|return|>,
    <|end|>, <|call|>, <|start|> or <|endoftext|> (or all of it) goes to the matcher.
  For the Qwen3 and GPT-OSS models the paper's completions are the generated token ids decoded with
  skip_special_tokens=False, so these markers are visible to the adapter. The format is chosen from the model id
  (--answer-format overrides it). A reasoning-on variant (--reasoning: the text after the last </think>, or the
  Harmony final channel) is included but was not used in the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from collections import Counter
from functools import lru_cache

from . import completions_io as io_  # noqa: E402

# =======================================================================================
# THE MATCHER (copied unchanged; grader_hash() below must return bf137bfc).
# =======================================================================================
# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def norm(text):
    """Matching normalisation: case-fold, underscores -> spaces, collapse whitespace, strip.

    Nothing else. No hyphen folding, no plural stripping, no synonyms, no edit distance.
    """
    return " ".join(text.replace("_", " ").split()).casefold()


def normalize_completion(text):
    """Cache-key normalisation: canonicalise line endings and strip the outer whitespace.

    Deliberately conservative. The key must never merge two completions that could get
    different verdicts, and `norm()` would do exactly that: it flattens `green_drawer_2`
    and `green drawer 2` onto the same string although the first cannot match a choice and
    the second can. Leading/trailing whitespace and \\r are the only things that provably
    cannot change a verdict, so they are the only things folded here.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def completion_key(text):
    return hashlib.sha256(normalize_completion(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the record's own choice list
# ---------------------------------------------------------------------------

class ChoiceSet:
    """Everything the matcher may know about a question: the 15 options it was offered.

    Built from the record's own `choices` string. Note what is NOT here: the gold answer,
    the question, the story, the prompt, and any global container vocabulary.
    """
    __slots__ = ("letters", "by_norm", "colours", "arities", "max_atoms", "names")

    def __init__(self, letters, by_norm, colours, arities, names):
        self.letters = letters          # casefolded letter -> canonical name
        self.by_norm = by_norm          # norm(name)        -> canonical name
        self.colours = colours          # first token of a name -> True
        self.arities = arities          # token counts seen among the names, e.g. {2, 3}
        self.max_atoms = max(arities)
        self.names = names              # canonical names, in offer order


@lru_cache(maxsize=4096)
def parse_choices(choices):
    """'A. blue_drawer, B. green_crate, ...' -> ChoiceSet."""
    letters, by_norm, colours, arities, names = {}, {}, {}, set(), []
    for item in choices.split(","):
        item = item.strip()
        if not item:
            continue
        head, sep, name = item.partition(".")
        head, name = head.strip(), name.strip()
        if not sep or len(head) != 1 or not head.isalpha() or not name:
            raise ValueError(f"unparsable choice item: {item!r}")
        key = norm(name)
        letters[head.casefold()] = name
        by_norm[key] = name
        tokens = key.split(" ")
        colours[tokens[0]] = True
        arities.add(len(tokens))
        names.append(name)
    if not names:
        raise ValueError("empty choice list")
    return ChoiceSet(letters, by_norm, colours, arities, names)


# ---------------------------------------------------------------------------
# tokenising  (hand-written: no regex in this file, so "no regex search" is checkable)
# ---------------------------------------------------------------------------

def _is_word_char(ch):
    return ch.isalnum() or ch == "_"


def atomise(text):
    """-> [(start, end, atom)] where an atom is a maximal run of [alnum_].

    Underscores are inside atoms, which is what makes `green_drawer_2` a single atom that
    cannot be sliced down to `green_drawer`.
    """
    atoms, i, n = [], 0, len(text)
    while i < n:
        if _is_word_char(text[i]):
            j = i + 1
            while j < n and _is_word_char(text[j]):
                j += 1
            atoms.append((i, j, text[i:j]))
            i = j
        else:
            i += 1
    return atoms


# Decoration that may precede a leading answer label: "**K**", "- K.", "(K)", "> K".
_HEAD_NOISE = frozenset(" \t\r\n*#>\"'`-~([{|.")
# What may follow a leading bare letter for it to be a label rather than prose.
# Whitespace is NOT here on purpose: "I think" and "A person" must not be answers.
_LABEL_FOLLOWERS = frozenset(".):,;]}*|-")
# Punctuation that can bind a bare "answer" lead-in to a letter ("Answer: K", "Answer - K").
_LEADIN_PUNCT = frozenset(":-=>")
# Whitespace and the hyphens that a HYPHEN NEAR-MISS diagnostic tolerates (never a match).
_HYPHENS = frozenset("-‐‑‒–—")

# Phrases after which a single letter is an answer. `True` = the phrase must be bound to the
# letter by punctuation, so that prose like "answer a question" cannot become answer "A".
_LEADINS = (
    (("the", "correct", "answer", "is"), False),
    (("the", "final", "answer", "is"), False),
    (("the", "answer", "is"), False),
    (("my", "answer", "is"), False),
    (("answer", "is"), False),
    (("final", "answer"), True),
    (("correct", "answer"), True),
    (("answer", "choice"), True),
    (("answer",), True),
    (("option",), False),
    (("choice",), False),
)
_LEADIN_MAX = max(len(phrase) for phrase, _ in _LEADINS)


class Mention:
    """One answer-like thing found in the completion, with where it was found."""
    __slots__ = ("pos", "kind", "value", "surface")

    def __init__(self, pos, kind, value, surface):
        self.pos = pos
        self.kind = kind            # letter | name | off_menu_letter | off_menu_name
        self.value = value          # canonical choice name, or None when off-menu
        self.surface = surface      # what was actually written, normalised

    @property
    def identity(self):
        """What counts as 'a distinct container named' for the ambiguity flag."""
        return self.value if self.value is not None else ("?", self.surface)


def _letter_mention(pos, letter, cs, route_kind):
    name = cs.letters.get(letter.casefold())
    if name is None:
        return Mention(pos, "off_menu_letter", None, letter.casefold())
    return Mention(pos, route_kind, name, letter.casefold())


def _leading_letter(text, atoms, cs):
    """A label at the very start of the completion: 'K', 'K.', '(K)', '**K**', '- K.', 'K\\n...'.

    The follower is looked for on the FIRST LINE only: a bare letter alone on its own line
    is a label, because the line break delimits it just as a '.' would. Whitespace is not a
    follower within a line, which is what stops "I think ..." and "A person ..." from being
    read as answers I and A.
    """
    if not atoms:
        return None
    start, end, atom = atoms[0]
    for ch in text[:start]:
        if ch not in _HEAD_NOISE:
            return None
    if len(atom) != 1 or not atom.isalpha():
        return None
    newline = text.find("\n", end)
    rest = text[end:len(text) if newline < 0 else newline].lstrip()
    if rest and rest[0] not in _LABEL_FOLLOWERS:
        return None
    return _letter_mention(start, atom, cs, "letter")


def _leadin_letters(text, atoms, cs):
    """Letters introduced explicitly: 'Answer: K', 'The answer is K', 'option K'."""
    out = []
    folded = [a[2].casefold() for a in atoms]
    for i in range(len(atoms)):
        for phrase, needs_punct in _LEADINS:
            j = i + len(phrase)
            if j >= len(atoms):
                continue
            if tuple(folded[i:j]) != phrase:
                continue
            letter = atoms[j][2]
            if len(letter) != 1 or not letter.isalpha():
                break
            if needs_punct:
                gap = text[atoms[j - 1][1]:atoms[j][0]]
                if not any(ch in _LEADIN_PUNCT for ch in gap):
                    break
            out.append(_letter_mention(atoms[j][0], letter, cs, "letter_leadin"))
            break
    return out


def _joined(text, atoms, i, m):
    """True if atoms[i:i+m] are separated by whitespace only (never by a hyphen or a '.')."""
    for k in range(i, i + m - 1):
        for ch in text[atoms[k][1]:atoms[k + 1][0]]:
            if not ch.isspace():
                return False
    return True


def _phrase(atoms, i, m):
    return norm(" ".join(atoms[k][2] for k in range(i, i + m)))


def _name_mentions(text, atoms, cs):
    """Container names, by exact normalised equality against this record's choice list.

    Longest window first (maximal munch), then non-overlapping left to right. A window that
    is not a choice but is container-SHAPED for this record -- it starts with one of this
    record's own colour words and has one of this record's own token counts -- is recorded as
    an off-menu mention, which is a REJECT with its own bucket. There is deliberately no
    global container list to consult: a name that is not on offer here is simply wrong.
    """
    out, i, n = [], 0, len(atoms)
    while i < n:
        hit = None
        for m in range(min(cs.max_atoms, n - i), 0, -1):
            if not _joined(text, atoms, i, m):
                continue
            phrase = _phrase(atoms, i, m)
            name = cs.by_norm.get(phrase)              # exact equality, not containment
            if name is not None:
                hit = (m, "name", name, phrase)
                break
        if hit is None:
            # SHORTEST window first here, the opposite of the exact pass above. Maximal munch
            # is right for an exact match (the longest exact name is the intended one) and
            # wrong for this fallback: a greedy off-menu span can swallow the first word of
            # the valid name that follows it ("red_herring green drawer" would eat "green").
            for m in range(1, min(cs.max_atoms, n - i) + 1):
                if not _joined(text, atoms, i, m):
                    continue
                phrase = _phrase(atoms, i, m)
                tokens = phrase.split(" ")
                if len(tokens) in cs.arities and cs.colours.get(tokens[0]) is not None:
                    hit = (m, "off_menu_name", None, phrase)
                    break
        if hit is None:
            i += 1
            continue
        m, kind, value, surface = hit
        out.append(Mention(atoms[i][0], kind, value, surface))
        i += m
    return out


def _hyphen_near_miss(text, atoms, cs):
    """Diagnostic only: would a hyphen-joined pair have matched, had hyphens been folded?

    Never affects a verdict. It exists so the report can say whether refusing to fold
    hyphens cost anything measurable.
    """
    n = len(atoms)
    for i in range(n):
        for m in range(2, min(cs.max_atoms, n - i) + 1):
            saw_hyphen = False
            ok = True
            for k in range(i, i + m - 1):
                for ch in text[atoms[k][1]:atoms[k + 1][0]]:
                    if ch in _HYPHENS:
                        saw_hyphen = True
                    elif not ch.isspace():
                        ok = False
                        break
                if not ok:
                    break
            if ok and saw_hyphen and cs.by_norm.get(_phrase(atoms, i, m)) is not None:
                return True
    return False


# ---------------------------------------------------------------------------
# extraction  --  completion and offered choices only, never a record, never the gold
# ---------------------------------------------------------------------------

def analyse(completion, choices):
    """Everything the matcher can say about one completion, WITHOUT knowing the answer.

    Returns a plain dict. `value` is the extracted canonical choice name or None; every
    None is a REJECT. Both arguments are the model's own output and the options the model
    was offered -- there is no third argument, so the gold answer cannot be consulted here.
    """
    text = normalize_completion(completion)
    if not text:
        return {"value": None, "route": "empty", "surface": "", "n_mentions": 0, "n_valid": 0,
                "last_value": None, "last_route": "empty", "ambiguous": False,
                "ambiguous_full": False, "flags": []}

    cs = parse_choices(choices)
    atoms = atomise(text)

    mentions = _name_mentions(text, atoms, cs)
    lead = _leading_letter(text, atoms, cs)
    if lead is not None:
        mentions.append(lead)
    mentions.extend(_leadin_letters(text, atoms, cs))
    # Letters sort before names at the same offset; a label IS the answer it labels.
    rank = {"letter": 0, "letter_leadin": 0, "off_menu_letter": 0, "name": 1, "off_menu_name": 1}
    mentions.sort(key=lambda mention: (mention.pos, rank[mention.kind]))

    if not mentions:
        return {"value": None, "route": "none", "surface": "", "n_mentions": 0, "n_valid": 0,
                "last_value": None, "last_route": "none", "ambiguous": False,
                "ambiguous_full": False,
                "flags": ["hyphen_near_miss"] if _hyphen_near_miss(text, atoms, cs) else []}

    # THE SELECTION RULE: the first mention that is a valid choice for THIS record. Tokens not
    # on offer are skipped, never rescued. When nothing on offer was named, the first
    # answer-shaped token still decides the REJECT bucket, so `not_a_choice` stays distinct
    # from `no_answer`.
    valid = [mention for mention in mentions if mention.value is not None]
    first = valid[0] if valid else mentions[0]
    last = valid[-1] if valid else mentions[-1]

    newline = text.find("\n")
    region_end = len(text) if newline < 0 else newline
    region = [mention for mention in mentions if mention.pos < region_end]

    # Distinct VALID choices only: an off-menu token is not a competing answer.
    distinct_region = {mention.value for mention in region if mention.value is not None}
    distinct_full = {mention.value for mention in valid}

    flags = []
    letter_values = {m.value for m in region
                     if m.kind in ("letter", "letter_leadin") and m.value is not None}
    name_values = {m.value for m in region if m.kind == "name" and m.value is not None}
    if letter_values and name_values and letter_values != name_values:
        flags.append("letter_name_conflict")
    if len({m.value for m in region if m.kind == "name" and m.value is not None}) >= 3:
        # the model echoed the offered choice list before answering
        flags.append("choice_list_echo")
    if valid and mentions[0].value is None:
        # the selection rule fired: an off-menu token was skipped to reach a valid choice
        flags.append("off_menu_first_then_valid")
    if _hyphen_near_miss(text, atoms, cs):
        flags.append("hyphen_near_miss")

    return {"value": first.value, "route": first.kind, "surface": first.surface,
            "n_mentions": len(mentions), "n_valid": len(valid),
            "last_value": last.value, "last_route": last.kind,
            "ambiguous": len(distinct_region) > 1, "ambiguous_full": len(distinct_full) > 1,
            "flags": flags}


def extract(completion, choices):
    """-> (value | None, route).

    THE SIGNATURE IS THE GUARANTEE: the completion and the offered choice list, nothing
    else. No record, no gold answer, no prompt. `--selftest` asserts these parameter names.
    """
    found = analyse(completion, choices)
    return found["value"], found["route"]


def grade(extracted, gold):
    """-> bool. REJECT short-circuits before any comparison exists."""
    if extracted is None:
        return False
    return norm(extracted) == norm(gold)


def bucket_for(route, passed):
    if route == "empty":
        return "empty"
    if route == "none":
        return "no_answer"
    if route in ("off_menu_letter", "off_menu_name"):
        return "not_a_choice"
    return "correct" if passed else "wrong_valid_choice"


BUCKETS = ("correct", "wrong_valid_choice", "not_a_choice", "no_answer", "empty")


# ---------------------------------------------------------------------------
# contract hash -- a matcher change invalidates cached verdicts instead of freezing them in
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def grader_hash():
    parts = []
    for fn in (norm, normalize_completion, parse_choices.__wrapped__, atomise,
               _leading_letter, _leadin_letters, _joined, _phrase, _name_mentions,
               analyse, extract, grade, bucket_for):
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):
            parts.append(f"<unreadable {fn!r}>")
    parts.append(repr(sorted(_HEAD_NOISE)))
    parts.append(repr(sorted(_LABEL_FOLLOWERS)))
    parts.append(repr(sorted(_LEADIN_PUNCT)))
    parts.append(repr(_LEADINS))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:8]


def verdict_for(completion, choices, gold):
    """One completion -> one verdict dict; `pass` is the verdict, `bucket` its outcome class."""
    started = time.monotonic()
    found = analyse(completion, choices)
    passed = grade(found["value"], gold)          # gold enters here and nowhere earlier
    bucket = bucket_for(found["route"], passed)
    if bucket == "correct":
        reason = f"PASS correct {found['value']}"
    elif bucket == "wrong_valid_choice":
        reason = f"FAIL wrong_valid_choice {found['value']}"
    elif bucket == "not_a_choice":
        reason = f"FAIL not_a_choice {found['route']}:{found['surface']}"
    else:
        reason = f"FAIL {bucket}"
    return {"pass": passed, "reason": reason, "bucket": bucket, "value": found["value"],
            "route": found["route"], "surface": found["surface"],
            "n_mentions": found["n_mentions"], "n_valid": found["n_valid"],
            "last_value": found["last_value"],
            "ambiguous": found["ambiguous"], "ambiguous_full": found["ambiguous_full"],
            "flags": found["flags"], "secs": round(time.monotonic() - started, 6),
            "grader": grader_hash()}



# =======================================================================================
# THE ANSWER ADAPTER (copied unchanged): completion text only; no choices, no answer key.
# =======================================================================================
ADAPTER_VERSION = "tom-reasoning-adapter-1"
QWEN_THINK_CLOSE = "</think>"
QWEN_EOS = ("<|im_end|>", "<|endoftext|>")
HARMONY_FINAL = "<|channel|>final"
HARMONY_MESSAGE = "<|message|>"
HARMONY_STOPS = ("<|return|>", "<|end|>", "<|call|>", "<|start|>", "<|endoftext|>")
FAMILY_ARMS = (("qwen3", "R"), ("qwen3", "M"), ("gptoss", "R"), ("gptoss", "M"))
REASONS = ("ok", "no_think_close", "no_final_channel", "final_header_cut")


def _strip_all(text, markers):
    """Remove every literal marker, to a fixed point (a removal cannot splice a new one)."""
    changed = True
    while changed:
        changed = False
        for marker in markers:
            if marker in text:
                text = text.replace(marker, "")
                changed = True
    return text


def _cut_at_first(text, stops):
    """Text up to the earliest occurrence of any stop marker, or all of it."""
    end = len(text)
    for stop in stops:
        at = text.find(stop)
        if 0 <= at < end:
            end = at
    return text[:end]


def answer_text(text, family, arm):
    """-> (answer | None, reason). THE SIGNATURE IS THE GUARANTEE: completion text only."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if family == "qwen3" and arm == "R":
        at = text.rfind(QWEN_THINK_CLOSE)
        if at < 0:
            return None, "no_think_close"
        return _strip_all(text[at + len(QWEN_THINK_CLOSE):], QWEN_EOS), "ok"
    if family == "qwen3" and arm == "M":
        return _strip_all(text, QWEN_EOS), "ok"
    if family == "gptoss" and arm == "R":
        head = text.find(HARMONY_FINAL)
        if head < 0:
            return None, "no_final_channel"
        body = text.find(HARMONY_MESSAGE, head + len(HARMONY_FINAL))
        if body < 0:
            return None, "final_header_cut"
        return _cut_at_first(text[body + len(HARMONY_MESSAGE):], HARMONY_STOPS), "ok"
    if family == "gptoss" and arm == "M":
        return _cut_at_first(text, HARMONY_STOPS), "ok"
    raise ValueError(f"unknown family/arm {family!r}/{arm!r}")


def verdict(text, family, arm, choices, gold):
    """One completion -> one verdict dict. Gold enters here and nowhere earlier. (The production adapter called the
    matcher module's verdict_for, which is defined above in this file.)"""
    answer, reason = answer_text(text, family, arm)
    if answer is None:
        return {"pass": False, "bucket": "no_answer", "adapter_reason": reason, "value": None,
                "route": "adapter", "reason": f"FAIL no_answer adapter:{reason}"}
    out = verdict_for(answer, choices, gold)
    out["adapter_reason"] = "ok"
    return out


# =======================================================================================
# Everything below drives the matcher (added for this release).
# =======================================================================================
HITOM_COMMIT = "459784fa8de09f35e9110e1e9454b41cd9b4f985"
HITOM_URL = (f"https://raw.githubusercontent.com/ying-hui-he/Hi-ToM_dataset/{HITOM_COMMIT}/"
             f"Hi-ToM_data/Hi-ToM_data.json")
HITOM_SHA256 = "088b6c54d21d1a0d3045812f36ac8b5b342346a6a996c003738976730ac256f4"   # 5,134,393 bytes, 1,200 records
MATCHER_CONTRACT = "bf137bfc"          # grader_hash() of the matcher that graded the paper's samples
ANSWER_FORMATS = ("plain", "qwen3", "gptoss")
_KEEP = ("choices", "answer", "question_order", "deception", "story_length", "prompting_type", "sample_id")


def question_id(record):
    """The question id the paper uses: s{sample_id}_o{question_order}_{prompting_type}."""
    return f"s{record['sample_id']}_o{record['question_order']}_{record['prompting_type']}"


def load_answers(path):
    """id -> option list and reference answer, from Hi-ToM_data.json.

    Only the option list, the answer and the factorial labels are kept; the story and the prompt are dropped as the
    file is read, so they can never reach the matcher. Every answer must be one of its question's own options."""
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)["data"]
    choices, gold, meta, broken = {}, {}, {}, []
    for raw in records:
        record = {k: raw.get(k) for k in _KEEP}
        del raw
        qid = question_id(record)
        if qid in choices or any(b[0] == qid for b in broken):
            raise SystemExit(f"{path}: duplicate question id {qid}")
        offered = record["choices"]
        answer = record["answer"]
        try:
            parsed = parse_choices(offered)
        except ValueError as error:
            broken.append((qid, str(error)))
            continue
        if parsed.by_norm.get(norm(answer)) is None:
            broken.append((qid, "gold answer is not one of the offered choices"))
            continue
        choices[qid] = offered
        gold[qid] = answer
        meta[qid] = {key: record.get(key) for key in
                     ("question_order", "deception", "story_length", "prompting_type")}
    return choices, gold, meta, broken


def answer_format_for(model):
    """The answer adapter the paper used for this model: gptoss for GPT-OSS, qwen3 for Qwen3, plain otherwise."""
    m = model.lower()
    if "gpt-oss" in m or "gpt_oss" in m:
        return "gptoss"
    if "qwen3" in m:
        return "qwen3"
    return "plain"


def grade_completion(text, answer_format, arm, choices, gold):
    """One completion -> verdict dict, exactly the paper's call: the matcher alone for plain text, else the adapter."""
    if answer_format == "plain":
        return verdict_for(text, choices, gold)
    return verdict(text, answer_format, arm, choices, gold)


def _grade_job(job):
    """job = (choices, gold, completions, finish_reasons, answer_format, arm) -> per-sample bits and buckets.
    Identical texts within a cell are graded once (the matcher is deterministic); every sample is counted."""
    choices, gold, texts, fins, fmt, arm = job
    memo, bits, buckets, reasons = {}, [], [], Counter()
    for text in texts:
        v = memo.get(text)
        if v is None:
            r = grade_completion(text, fmt, arm, choices, gold)
            if not isinstance(r["pass"], bool):
                raise RuntimeError("the matcher returned a non-bool verdict")
            v = memo[text] = (r["pass"], r["bucket"], r.get("adapter_reason"))
        bits.append(v[0])
        buckets.append(v[1])
        if v[2] is not None:
            reasons[v[2]] += 1
    trunc = [f == "length" for f in fins] if fins is not None else None
    return bits, buckets, trunc, reasons, len(memo)


def grade_inputs(paths, choices, gold, jobs=1, model=None, answer_format="auto", arm="M"):
    """Grade every cell under `paths` -> (CountsTable, Summary, totals)."""
    import concurrent.futures as cf
    counts = io_.CountsTable()
    summary = io_.Summary()
    totals = Counter()
    adapter_reasons = Counter()
    fmts = Counter()
    with cf.ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        pending = {}

        def drain(block):
            done, _ = cf.wait(list(pending), return_when=cf.ALL_COMPLETED if block else cf.FIRST_COMPLETED)
            for fut in done:
                cell = pending.pop(fut)
                bits, buckets, trunc, reasons, distinct = fut.result()
                c = sum(bits)
                counts.add(cell, len(bits), c, texts_hash=cell["_hash"])
                nt = sum(trunc) if trunc else 0
                ntp = sum(1 for b, t in zip(bits, trunc) if b and t) if trunc else 0
                summary.add(cell, Counter(buckets), nt, ntp)
                adapter_reasons.update(reasons)
                totals.update({"cells": 1, "samples": len(bits), "passes": c, "distinct_texts": distinct})

        for cell in io_.iter_cells(paths, "hitom", model=model):
            qid = cell["id"]
            if qid not in gold:
                raise SystemExit(f"{cell['source']}: {qid!r} is not a Hi-ToM question id")
            fmt = answer_format_for(cell["model"]) if answer_format == "auto" else answer_format
            fmts[f"{cell['model']} -> {fmt}"] += 1
            cell["_hash"] = io_.texts_sha256(cell["completions"])
            fut = ex.submit(_grade_job, (choices[qid], gold[qid], cell["completions"], cell["finish_reasons"],
                                         fmt, arm))
            cell.pop("completions")
            pending[fut] = cell
            if len(pending) >= 2 * max(1, jobs):
                drain(False)
        while pending:
            drain(True)
    totals["adapter_reasons"] = dict(adapter_reasons)
    totals["answer_formats"] = dict(fmts)
    return counts, summary, totals


# --------------------------------------------------------------------------- self-test
_SELFTEST_CHOICES = "A. violet_box, B. amber_crate, C. teal_drawer, D. teal_drawer_2, E. amber_basket"   # fictional


def structural_checks():
    """The guarantees that keep the prompt and the answer key out of the extraction step."""
    out = []
    for fn, want in ((extract, ("completion", "choices")), (analyse, ("completion", "choices")),
                     (grade, ("extracted", "gold")), (answer_text, ("text", "family", "arm"))):
        got = tuple(inspect.signature(fn).parameters)
        out.append((f"{fn.__name__} signature is {want}", got == want, str(got)))
    out.append(("grade(None, gold) rejects", all(grade(None, g) is False for g in ("violet_box", "", "None")), ""))
    out.append(("matcher contract hash is the paper's", grader_hash() == MATCHER_CONTRACT, grader_hash()))
    out.append(("module never imports re", "re" not in globals(), ""))
    cases = (("C. teal_drawer", "C", "correct"), ("teal drawer 2", "C", "wrong_valid_choice"),
             ("The answer is E.", "C", "wrong_valid_choice"), ("It is in the teal cupboard.", "C", "not_a_choice"),
             ("I do not know.", "C", "no_answer"), ("   ", "C", "empty"))
    for text, letter, want in cases:
        cs = parse_choices(_SELFTEST_CHOICES)
        r = verdict_for(text, _SELFTEST_CHOICES, cs.letters[letter.casefold()])
        out.append((f"synthetic case {text!r}", r["bucket"] == want, r["bucket"]))
    return out


def selftest(stream=sys.stdout):
    ok = True
    for label, passed, detail in structural_checks():
        print(f"  [{'ok' if passed else 'FAIL'}] {label:48s} {detail}", file=stream)
        ok = ok and passed
    print("SELFTEST " + ("OK" if ok else "FAILED"), file=stream)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", help="completions files or directories (*.jsonl, *.jsonl.gz)")
    ap.add_argument("--out", help="counts.jsonl to write")
    ap.add_argument("--summary", help="optional JSON: buckets and truncation per (model, setting), counts only")
    ap.add_argument("--dataset", help="local copy of Hi-ToM_data.json (default: download and cache it)")
    ap.add_argument("--cache-dir", default=io_.default_cache_dir(), help="where downloaded data is kept")
    ap.add_argument("--no-verify-data", action="store_true", help="skip the sha256 check of the dataset")
    ap.add_argument("--model", help="model id for lines that do not name one")
    ap.add_argument("--answer-format", choices=("auto",) + ANSWER_FORMATS, default="auto",
                    help="answer adapter; auto = from the model id, as in the paper")
    ap.add_argument("--reasoning", action="store_true", help="reasoning-on adapter (not used in the paper)")
    ap.add_argument("--jobs", type=int, default=1, help="grading processes (default 1)")
    ap.add_argument("--force", action="store_true", help="overwrite --out / --summary")
    ap.add_argument("--selftest", action="store_true", help="structural checks and synthetic cases, then exit")
    ap.add_argument("--try", dest="try_text", metavar="COMPLETION", help="grade one completion for --qid")
    ap.add_argument("--qid", help="question id for --try")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if grader_hash() != MATCHER_CONTRACT:
        raise SystemExit(f"the matcher's contract hash is {grader_hash()}, not the paper's {MATCHER_CONTRACT}: its "
                         f"code was changed, so its verdicts are not the paper's; refusing to grade")
    out_path = summary_path = None
    if a.try_text is None:
        if not a.inputs or not a.out:
            ap.error("--inputs and --out are required")
        out_path, summary_path = io_.check_outputs([a.out, a.summary], force=a.force)     # before any work
    path = io_.fetch_pinned(HITOM_URL, HITOM_SHA256, a.cache_dir, "Hi-ToM_data.json", a.dataset,
                            check=not a.no_verify_data)
    choices, gold, meta, broken = load_answers(path)
    if broken:
        raise SystemExit(f"{len(broken)} unusable records in {path}, e.g. {broken[:2]}")
    arm = "R" if a.reasoning else "M"
    if a.try_text is not None:
        if a.qid not in gold:
            ap.error("--try needs a valid --qid")
        if a.answer_format != "auto":
            fmt = a.answer_format
        else:
            fmt = answer_format_for(a.model) if a.model else "plain"
        r = grade_completion(a.try_text, fmt, arm, choices[a.qid], gold[a.qid])
        print(json.dumps({k: r[k] for k in ("pass", "bucket", "route", "value") if k in r}))
        return 0
    counts, summary, totals = grade_inputs(a.inputs, choices, gold, jobs=a.jobs, model=a.model,
                                           answer_format=a.answer_format, arm=arm)
    rows = counts.sorted_rows()
    io_.write_jsonl_atomic(out_path, rows, force=a.force)
    for w in counts.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if summary_path:
        io_.write_json_atomic(summary_path, {
            "benchmark": "hitom", "matcher_contract": grader_hash(), "adapter_version": ADAPTER_VERSION,
            "reasoning_arm": arm, "dataset": {"url": HITOM_URL, "sha256": io_.sha256_file(path),
                                              "verified": not a.no_verify_data},
            "python": sys.version.split()[0], "totals": dict(totals), "by_setting": summary.as_json()},
            force=a.force)
    print(f"{len(rows)} cells, {totals['samples']} samples, {totals['passes']} correct -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
