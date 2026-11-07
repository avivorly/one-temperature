#!/usr/bin/env python3
"""Step 2 for MATH-500: grade completions and write the counts file.

    python grade_math500.py --gen-dir runs/mistral7b_math500 --dataset math500_test.jsonl --out counts.jsonl

--dataset is the public test split of HuggingFaceH4/MATH-500 saved as JSONL (fields unique_id, answer); without it
the split is loaded from the Hugging Face hub. Output: one line per (question, temperature)
    {"id", "temperature", "n", "c", "c_strict"}
c counts completions graded correct by the full grader, c_strict by the exact layer alone. Every completion counts
once (instance-weighted); verdicts are cached by the extracted answer string only to save time.

The grader below is the frozen grader of the paper's MATH-500 runs (version v2.5), copied unchanged except for
comments. Its design:
  * extract_boxed(completion) takes the completion ONLY; the LAST \\boxed{} span is the answer; a completion with no
    \\boxed{} is its own bucket (no_boxed) and fails.
  * normalize() both sides; the exact layer is exact normalised equality, never substring or regex search.
  * Then an equivalence layer of generic rules (never per question), and SymPy equivalence through parse_expr with a
    restricted namespace and empty __builtins__ -- never sympify/eval on model output.
  * Every comparison is time-bound (SIGALRM, Unix main thread); a timeout or an internal error REJECTS.
  * Only a real bool from SymPy is believed.
Rule codes recorded on each verdict:
  R1 units       degree marks and ONE trailing unit block (\\text{..}, \\mbox{..}, \\mathrm{..}) are dropped from both sides.
  R2 prefix      a single-letter `v =` prefix is dropped once; on both sides the letters must match.
  R3 lists       a gold that is a bare comma list or a set is a multiset; tuples, intervals, vectors, matrices stay ordered.
  R4 \\pm         `a \\pm b` is the multiset {a+b, a-b}.
  R5 fail closed any LaTeX command the translator does not handle makes the pair `unparseable` (a fail); a fixed
                 whitelist of pure layout commands is stripped first.
  R6 buckets     timeout -> grader_timeout, internal error -> grader_error; both fail and both are counted.
  MX             a gold that is a mixed number (2\\frac{1}{3}) means addition; a candidate's own mixed number must match
                 under both the product and the school reading.
  TS thousands   one comma-grouped number (12,345) is that number, only against a single number; never against a list.
  BE exponents   ^{t} is ^t for one letter or digit t.
  PX prefixes    a subscripted or primed single letter, or an uppercase segment name, before '='; `v \\in` before a set.
  SS singleton   a one-element set against a scalar gold is that element.
  TP tuples      a bare comma list against a parenthesised tuple gold of the same length, element by element in order.
Requires sympy (see requirements.txt).
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
import re
import signal
from contextlib import contextmanager

GRADER_VERSION = "v2.5"

# ---------------------------------------------------------------------------------------
# Extraction. Takes a completion string and NOTHING else.               (verbatim v1)
# ---------------------------------------------------------------------------------------


def extract_boxed(completion):
    """Return (last_boxed_content | None, n_spans). Sees only the model's own text."""
    if not isinstance(completion, str):
        return None, 0
    spans = []
    i = 0
    while True:
        j = completion.find("\\boxed", i)
        if j == -1:
            break
        k = completion.find("{", j)
        if k == -1:
            i = j + 6
            continue
        depth, m = 1, k + 1
        while m < len(completion) and depth:
            if completion[m] == "{":
                depth += 1
            elif completion[m] == "}":
                depth -= 1
            m += 1
        if depth == 0:
            spans.append(completion[k + 1:m - 1])
        i = m if m > j else j + 6
    if not spans:
        return None, 0
    return spans[-1], len(spans)          # LAST span, decided in advance


# ---------------------------------------------------------------------------------------
# Normalisation                                                           (verbatim v1)
# ---------------------------------------------------------------------------------------
_LEFTRIGHT = re.compile(r"\\(?:left|right|!|,|;|\s)")


def normalize(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = _LEFTRIGHT.sub("", s)
    s = s.replace("\\dots", "").replace("\\ldots", "").replace("\\cdot", "*")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    if s.endswith("^\\circ"):
        s = s[:-6]
    # \frac12 -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{([^{}]+)\}", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


# ---------------------------------------------------------------------------------------
# Timeout. A timeout REJECTS (R6: as grader_timeout).
# ---------------------------------------------------------------------------------------
class GraderTimeout(Exception):
    pass


class GraderInternalError(Exception):
    """R6: the grader itself failed (resource exhaustion, an exception after parsing). Never a verdict."""


_ALARM = {"fired": False}


@contextmanager
def time_limit(seconds):
    """v1's SIGALRM bound, hardened in one way: after the deadline the alarm RE-FIRES every 50 ms, and a flag records
    that it fired. sympy has internal `except Exception` blocks; one of them can swallow a single GraderTimeout and keep
    computing. With the re-fire the next bytecode boundary raises again, and grade_one discards any verdict reached after
    the flag was set (fail closed)."""
    def handler(signum, frame):
        _ALARM["fired"] = True
        raise GraderTimeout()
    _ALARM["fired"] = False
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds, 0.05)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


TIMEOUT_S = 2.0       # v1's bound per comparison; a multiset gets TIMEOUT_S per gold item

# ---------------------------------------------------------------------------------------
# Equivalence layer — parse_expr with a restricted namespace, NEVER sympify/eval.   (verbatim v1)
# ---------------------------------------------------------------------------------------
_SAFE = None


def _safe_ns():
    global _SAFE
    if _SAFE is None:
        import sympy
        _SAFE = {
            "pi": sympy.pi, "E": sympy.E, "I": sympy.I,
            "sqrt": sympy.sqrt, "Abs": sympy.Abs, "log": sympy.log, "ln": sympy.log,
            "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
            "exp": sympy.exp, "frac": sympy.Rational,
        }
    return _SAFE


_GLOBALS = None


def _safe_globals():
    """Globals for parse_expr's internal eval (v1, verbatim).

    `__builtins__` is set explicitly to an empty dict: Python's `eval` INJECTS the real builtins into globals when the
    key is absent, so an empty global_dict is not a sandbox.
    """
    global _GLOBALS
    if _GLOBALS is None:
        import sympy
        _GLOBALS = {
            "__builtins__": {},
            "Integer": sympy.Integer, "Float": sympy.Float,
            "Rational": sympy.Rational, "Symbol": sympy.Symbol,
        }
    return _GLOBALS


_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")


def _var_letters(s):
    """Letters that could become free symbols, ignoring LaTeX command names (v1, verbatim). Two expressions built
    from different variables cannot be equal; the check runs on the strings, before parse_expr, because
    implicit-multiplication parsing of many digit-letter juxtapositions is itself the explosion."""
    return set(re.sub(r"[^A-Za-z]", "", _LATEX_CMD.sub(" ", s or "")))


def _translate(s):
    """v1's _to_expr translations, verbatim, MINUS its last step `re.sub(r"\\\\[a-zA-Z]+", "", t)` (R5)."""
    t = s
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
    t = t.replace("\\pi", "pi").replace("^", "**")
    t = t.replace("\\times", "*").replace("\\div", "/")
    return t


def _parse(t):
    """parse_expr on the restricted namespace. None = not parseable (as v1). R6: a timeout propagates; resource
    exhaustion (MemoryError, RecursionError) is an internal error, not a parse failure."""
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
        implicit_multiplication_application
    if not t or not re.fullmatch(r"[0-9a-zA-Z_+\-*/(). ]*", t):
        return None
    try:
        e = parse_expr(
            t, local_dict=_safe_ns(), global_dict=_safe_globals(), evaluate=True,
            transformations=standard_transformations + (implicit_multiplication_application,))
    except GraderTimeout:
        raise
    except (MemoryError, RecursionError) as ex:
        raise GraderInternalError(f"parse: {type(ex).__name__}") from ex
    except Exception:
        return None
    # A bare function name (`sqrt`, `log`) parses to a function, not an expression: not a value.
    return e if isinstance(e, sympy.Expr) else None


MATCH, NO, UNPARSEABLE = "match", "no", "unparseable"


# v2.3: a mixed number -- an integer written directly before \frac{int}{int}. v1's translation reads it as a product.
_MIXED = re.compile(r"(?<![\d.])(\d+)\\frac\{(-?\d+)\}\{(\d+)\}")
_GOLD_MIXED = re.compile(r"-?\d+\\frac\{\d+\}\{\d+\}")      # v2.4: a whole gold that is a mixed number (fullmatch)


def _school(s):
    """v2.3: the school reading of every mixed number in s, a\\frac{b}{c} -> (a+\\frac{b}{c})."""
    return _MIXED.sub(lambda m: "(" + m.group(1) + "+\\frac{" + m.group(2) + "}{" + m.group(3) + "})", s)


def _scalar(a, b):
    """One item against one item -> (MATCH | NO | UNPARSEABLE, reason). Exact string equality first (sound), then v1's
    sympy equivalence with R5 in front of it. Runs inside grade_one's time bound.

    v2.3: when either side holds a mixed number, the two sides must be equal under BOTH readings --
    v1's product a*b/c and the school reading a + b/c. Strictly stricter than v2.2: it can only remove a credit."""
    if not a or not b:
        return NO, "empty_item"                    # empty never matches empty
    if a == b:
        return MATCH, "equal_strings"
    if _MIXED.search(a) or _MIXED.search(b):
        st, why = _sympy_equal(a, b)
        if st != MATCH:
            return st, why
        st2, why2 = _sympy_equal(_school(a), _school(b))
        return (MATCH, why) if st2 == MATCH else ((st2, why2) if st2 == UNPARSEABLE else (NO, "mixed_number"))
    return _sympy_equal(a, b)


def _sympy_equal(a, b):
    """v1's sympy equivalence, with R5 in front of it -> (MATCH | NO | UNPARSEABLE, reason)."""
    import sympy
    ta, tb = _translate(a), _translate(b)
    if "\\" in ta or "\\" in tb:
        return UNPARSEABLE, "r5_command"           # R5: never delete what _to_expr does not translate
    if _var_letters(a) != _var_letters(b):
        return NO, "var_letters"
    ea, eb = _parse(ta), _parse(tb)
    if ea is None or eb is None:
        return NO, "parse_failed"
    # v1: different free symbols can never cancel, and simplify() of them may never return.
    if ea.free_symbols != eb.free_symbols:
        return NO, "free_symbols"
    try:
        d = sympy.simplify(ea - eb)
        v = (d == 0)
        # sympy truthiness: only a real bool is believed (v1).
        if not isinstance(v, bool):
            v = bool(d.is_zero) if d.is_zero is not None else False
    except GraderTimeout:
        raise
    except Exception as ex:
        raise GraderInternalError(f"compare: {type(ex).__name__}") from ex
    return (MATCH, "sympy_equal") if v else (NO, "sympy_unequal")


# ---------------------------------------------------------------------------------------
# R1-R4: the forms the two sides are compared in.
# ---------------------------------------------------------------------------------------
# R1, on normalized text (no spaces). v2.1: + a Unicode degree sign in any of its forms, and \degree.
_DEGREE = re.compile(r"\^(?:\\circ|\{\\circ\})|\\(?:text|mbox)\{°\}|\^\{°\}|\^°|°|\\degree(?![A-Za-z])")
_UNIT_TAIL = re.compile(r"\\(?:text|mbox)\{([^{}]*)\}(\^(?:\d|\{\d+\}))?$")
# v2.1: words may carry a leading hyphen and be joined by runs of . - /; or the block is a lone unit sign.
_UNIT_WORDS = re.compile(r"-?[A-Za-z]+(?:[.\-/]+[A-Za-z]+)*\.?|¢")
# A trailing \mathrm unit on the RAW string, before normalize() unwraps \mathrm{..} into bare letters. Single-letter
# \mathrm is math notation (i, e, d), so a \mathrm unit word needs >= 2 letters.
_MATHRM_TAIL = re.compile(
    r"\\mathrm\s*\{([^{}]*)\}((?:\s*\^\s*(?:\d|\{\s*\d+\s*\}))?(?:\s|\\[,;:! ]|\.|\$)*)$")
_PREFIX = re.compile(r"([A-Za-z])=(.+)", re.S)                           # R2, on normalized text
_PM = re.compile(r"\\(pm|mp)(?![A-Za-z])")                                 # R4
# v2.2 (b): a FIXED whitelist of pure layout/spacing commands, stripped on the RAW string -- before
# normalize() deletes whitespace and merges a command with the letters after it. \left, \right, \, \; \! and "\ "
# already go in normalize(). Nothing else is stripped: every other untranslated command still fails closed (R5).
_LAYOUT = re.compile(r"\\(?:displaystyle|textstyle|big|bigl|bigr|Big|Bigl|Bigr|bigg|biggl|biggr|quad|qquad)"
                     r"(?![A-Za-z])|\\:")
# R3: a comma-grouped number is a number, not a list. v2.5 TS: a fixed pattern, on a side's form ({,} read as a
# comma); it replaces v2.0's [+-]? pattern on the normalized gold, which no MATH-500 gold tells apart.
_GROUPED = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
# TS, on the RAW text: a comma followed by whitespace, ~ or a positive LaTeX space separates a LIST ("3, 100").
# normalize() deletes all of these, so only the raw text can tell. ",\!" and "{,}" group digits (MATH's own style).
_LISTSEP = re.compile(r",(?:\s|~|\\[ ,;:]|\\q?quad(?![A-Za-z]))")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")                                     # TS: "a single number"
# BE: ^{t} -> ^t for ONE letter or digit, never directly before a letter, digit or point (2^{3}4 is not 2^34).
_BRACED_EXP = re.compile(r"\^\{([A-Za-z0-9])\}(?![A-Za-z0-9.])")
# PX, on normalized text: a subscripted or primed single letter, or an uppercase segment name, before the only '='.
_PX_EQ = re.compile(r"([A-Za-z](?:_(?:\{[A-Za-z0-9]+\}|[A-Za-z0-9])|'+)|[A-Z]{2})=(.+)", re.S)
# PX: `<v> \in` before an interval or a set -- the rest opens with ( [ \{ and closes with ) ] \}.
_PX_IN = re.compile(r"([A-Za-z])\\in((?:\(|\[|\\\{).*(?:\)|\]|\\\}))", re.S)
_IN = re.compile(r"\\in(?![A-Za-z])")


def _protect_mathrm_unit(raw):
    """R1 for \\mathrm: rewrite a trailing `\\mathrm{<words>}` unit into \\text{} so normalize() keeps it visible."""
    m = _MATHRM_TAIL.search(raw)
    if m is None:
        return raw, False
    words = re.sub(r"[\s~]+", "", m.group(1))
    before = raw[:m.start()].strip(" \t\r\n\\,;:!$")
    if len(re.sub(r"[^A-Za-z]", "", words)) < 2 or not _UNIT_WORDS.fullmatch(words) or not before:
        return raw, False
    return raw[:m.start()] + "\\text{" + m.group(1) + "}" + m.group(2), True


def _strip_unit_tail(s):
    """R1: drop ONE trailing \\text/\\mbox unit block holding words, with its power, if a value comes before it."""
    m = _UNIT_TAIL.search(s)
    if m and m.start() > 0 and _UNIT_WORDS.fullmatch(m.group(1).replace("~", "")):   # v2.1: ~ is a space
        return s[:m.start()], True
    return s, False


def _r1_form(raw):
    """One side in the equivalence layer's form: the v2.2 layout whitelist on the RAW string, then v1 normalize() with
    R1 applied. Returns (form, r1_applied, layout_stripped)."""
    lay = _LAYOUT.sub(" ", raw)
    p, mrm = _protect_mathrm_unit(lay)
    s = normalize(p)
    t = _DEGREE.sub("", s)
    t, cut = _strip_unit_tail(t)
    return t, (mrm or cut or t != s), lay != raw


def _split_top(s):
    """Split at commas outside (), [], {} (\\{ and \\} count as braces). None if the brackets do not balance."""
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return None
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if depth != 0:
        return None
    parts.append("".join(cur))
    return parts


def _set_inner(s):
    """The inside of `\\{ ... \\}` when those braces enclose the whole string, else None."""
    if len(s) >= 4 and s.startswith("\\{") and s.endswith("\\}"):
        inner = s[2:-2]
        if _split_top(inner) is not None:
            return inner
    return None


def _grouped(s, raw):
    """v2.5 TS: s (one side's form) as ONE comma-grouped number without its separators, else None. `raw` is that side's
    raw text: a list separator anywhere in it means the commas are not grouping digits."""
    f = s.replace("{,}", ",")
    if _GROUPED.fullmatch(f) and not _LISTSEP.search(raw):
        return f.replace(",", "")
    return None


def _ts_list_clash(extracted, gold, ne):
    """v2.5 TS, at the exact layer: v1's normalize() deletes whitespace, so "3,100" (one grouped number) and "3, 100" (a
    list of two) are equal there. When exactly one raw side separates its commas and the shared form is a grouped
    number, the two answers differ. Linear string work only."""
    if (_LISTSEP.search(extracted) is None) == (_LISTSEP.search(gold) is None):
        return False
    return _GROUPED.fullmatch(_core(ne).replace("{,}", ",")) is not None


def _split_prefix(s):
    """R2 / v2.5 PX: a whole-answer prefix, stripped once -> (code, name, rest), or None.
    R2 exactly as v2.2: a single letter before the only '='. PX: a subscripted or primed single letter, or an uppercase
    segment name, before the only '=' with no \\in anywhere; or `<v> \\in` before an interval or a set, with no '=' and
    no other \\in."""
    if s.count("=") == 1:
        m = _PREFIX.fullmatch(s)
        if m:
            return "R2", m.group(1), m.group(2)
        m = None if _IN.search(s) else _PX_EQ.fullmatch(s)
        if m:
            return "PX", m.group(1), m.group(2)
    elif "=" not in s and len(_IN.findall(s)) == 1:
        m = _PX_IN.fullmatch(s)
        if m:
            return "PX", m.group(1) + "\\in", m.group(2)
    return None


def _core(s):
    """A side without its whole-answer prefix (R2 / PX), for the structure tests."""
    p = _split_prefix(s)
    return s if p is None else p[2]


def _tuple_parts(s):
    """v2.5 TP: the parts of a parenthesised tuple -- ONE ( .. ) pair around the whole string, >= 2 top-level parts --
    or None. A gold holding \\infty is an interval, never a tuple."""
    if len(s) < 2 or s[0] != "(" or s[-1] != ")" or "\\infty" in s:
        return None
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0 and i < len(s) - 1:
                return None                               # the first ( closes early: (a)\cup(b), (a)(b), ...
    parts = _split_top(s[1:-1])
    return parts if parts is not None and len(parts) >= 2 else None


def _gold_is_multiset(g, raw):
    """R3 / R4: is the gold a set, a bare comma list (not a comma-grouped number), or a +- item? v2.5: g comes without a
    whole-answer prefix, and the grouped-number test reads the raw text too (TS), so "3, 100" is a list."""
    if _set_inner(g) is not None:
        return "R3"
    parts = _split_top(g)
    if parts is not None and len(parts) >= 2 and _grouped(g, raw) is None:
        return "R3"
    if _PM.search(g):
        return "R4"
    return None


def _items(s, rules):
    """One side of a multiset comparison -> (items, prefix name, prefix code). Adds the rules it applies to `rules`."""
    letter = code = None
    p = _split_prefix(s)
    if p:                                                 # v2.2 (a) / v2.5 PX: a whole-answer prefix, strip once
        code, letter, s = p
        rules.add(code)
    inner = _set_inner(s)
    if inner is not None:
        elems = _split_top(inner)
    else:
        parts = _split_top(s)
        elems = parts if parts is not None and len(parts) >= 2 else [s]
    out = []
    for e in elems:
        e2, cut = _strip_unit_tail(e)                     # R1, per element
        if cut:
            rules.add("R1")
        out.append(e2)
    elems = out
    if letter is None and len(elems) >= 2:                # R2 per element: all carry the SAME letter, or none stripped
        ms = [_PREFIX.fullmatch(e) for e in elems]
        if all(ms) and len({m.group(1) for m in ms}) == 1:
            letter, code, elems = ms[0].group(1), "R2", [m.group(2) for m in ms]
            rules.add("R2")
    items = []
    for e in elems:                                       # R4: one +- per item expands to two values
        hits = _PM.findall(e)
        if len(hits) == 1:
            plus, minus = ("+", "-") if hits[0] == "pm" else ("-", "+")
            items.append(_PM.sub(lambda _: plus, e))
            items.append(_PM.sub(lambda _: minus, e))
            rules.add("R4")
        else:
            items.append(e)                               # 0 or >= 2 (cannot expand; R5 refuses the leftover)
    return items, letter, code


def _multiset(C, G):
    """Perfect matching of candidate items C onto gold items G under _scalar -> (MATCH | NO | UNPARSEABLE, reason)."""
    tainted = any("\\" in _translate(x) for x in C + G)    # some item the sympy layer refuses (R5)
    if len(C) != len(G):
        return (UNPARSEABLE, "r5_command") if tainted else (NO, "multiset_len")
    n = len(G)
    memo = {}

    def ok(i, j):
        if (i, j) not in memo:
            memo[(i, j)] = C[i] == G[j] or _scalar(C[i], G[j])[0] == MATCH
        return memo[(i, j)]

    # exact string pairs first, so a cheap match is never preceded by a slow sympy call
    for i in range(n):
        for j in range(n):
            if C[i] == G[j]:
                memo[(i, j)] = True
    owner = [-1] * n

    def augment(i, seen):
        for j in range(n):
            if j not in seen and ok(i, j):
                seen.add(j)
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in range(n):
        if not augment(i, set()):
            return (UNPARSEABLE, "r5_command") if tainted else (NO, "multiset_nomatch")
    return MATCH, "multiset_match"


def _ordered(C, G):
    """v2.5 TP: candidate item i against gold item i under _scalar, equal length -> (MATCH | NO | UNPARSEABLE, reason)."""
    tainted = any("\\" in _translate(x) for x in C + G)    # some item the sympy layer refuses (R5)
    if len(C) != len(G):
        return (UNPARSEABLE, "r5_command") if tainted else (NO, "tp_len")
    for c, g in zip(C, G):
        if not (c == g or _scalar(c, g)[0] == MATCH):
            return (UNPARSEABLE, "r5_command") if tainted else (NO, "tp_nomatch")
    return MATCH, "ordered_match"


def _prepare(extracted, gold):
    """The v2 equivalence layer, string work only (R1-R4, MX, TS, BE, PX, SS, TP; no sympy).
    -> (kind, a, b, rules, n_gold_items) with kind "multiset" (a, b = item lists), "ordered" (a, b = item lists, v2.5
    TP), "scalar" (a, b = strings) or "decided" (a = status, b = reason)."""
    rules = set()
    e, r1e, le = _r1_form(extracted)
    g, r1g, lg_ = _r1_form(gold)
    if r1e or r1g:
        rules.add("R1")
    if le or lg_:
        rules.add("R5")                                   # v2.2 (b): a whitelisted layout command was stripped
    e2, g2 = _BRACED_EXP.sub(r"^\1", e), _BRACED_EXP.sub(r"^\1", g)
    if (e2, g2) != (e, g):                                # v2.5 BE: ^{t} -> ^t, layout only
        rules.add("BE")
        e, g = e2, g2
    ce, cg = _core(e), _core(g)
    # v2.5 TS: a grouped-number candidate never meets a list, tuple, set or interval gold. TS does not apply there, and
    # R3 or TP would otherwise split it at its grouping commas. (The other way round nothing can credit it.)
    if _grouped(ce, extracted) is not None and _grouped(cg, gold) is None and "," in cg.replace("{,}", ","):
        rules.add("TS")
        return "decided", NO, "ts_vs_list", rules, 1
    mode = _gold_is_multiset(cg, gold)
    if mode is not None:
        rules.add(mode)
        G, lg, kg = _items(g, rules)
        C, lc, kc = _items(e, rules)
        if lg is not None and lc is not None and lg != lc:
            return "decided", NO, "r2_letters" if kg == kc == "R2" else "px_names", rules, len(G)
        return "multiset", C, G, rules, len(G)
    # scalar gold. R2 (v2.2) / PX (v2.5): a prefix is stripped once, only when it holds the string's only '=' (or \in)
    pe, pg = _split_prefix(e), _split_prefix(g)
    if pe or pg:
        rules.update(p[0] for p in (pe, pg) if p)
        if pe and pg and pe[1] != pg[1]:
            return "decided", NO, "r2_letters" if pe[0] == pg[0] == "R2" else "px_names", rules, 1
        e = pe[2] if pe else e
        g = pg[2] if pg else g
    if len(_PM.findall(e)) == 1:                          # R4: two values against a single gold
        rules.add("R4")
        return "decided", NO, "pm_vs_scalar", rules, 1
    # v2.5 SS: a set of ONE element against a scalar gold is that element.
    inner = _set_inner(e)
    if inner is not None and len(_split_top(inner)) == 1:
        e = inner
        rules.add("SS")
    # v2.5 TS: a comma-grouped number is that number without separators, ONLY against a single number.
    ge, gg = _grouped(e, extracted), _grouped(g, gold)
    if ge is not None or gg is not None:
        a = ge if ge is not None else (e if _NUMBER.fullmatch(e) else None)
        b = gg if gg is not None else (g if _NUMBER.fullmatch(g) else None)
        if a is not None and b is not None:
            e, g = a, b
            rules.add("TS")
    # v2.5 TP: a BARE comma list against a parenthesised tuple gold, element by element in order. A
    # prefixed list (x = a, b) lists values of x, not a pair.
    tp = _tuple_parts(g)
    if tp is not None and pe is None:
        parts = _split_top(e)
        if parts is not None and len(parts) >= 2:
            rules.add("TP")
            return "ordered", parts, tp, rules, len(tp)
    # v2.4: a GOLD that IS a mixed number means addition. A candidate that holds a mixed number of its
    # own keeps v2.3's both-readings rule in _scalar, so this applies only to a candidate without one.
    if _GOLD_MIXED.fullmatch(g) and not _MIXED.search(e):
        g = _school(g)
        rules.add("MX")
    return "scalar", e, g, rules, 1


def _decide(kind, a, b):
    """The sympy half of the equivalence layer -> (status, reason)."""
    if kind == "decided":
        return a, b
    if kind == "multiset":
        return _multiset(a, b)
    return _ordered(a, b) if kind == "ordered" else _scalar(a, b)


# ---------------------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------------------
BUCKETS = ("no_boxed", "exact_match", "equivalence_match", "parsed_wrong", "unparseable",
           "grader_timeout", "grader_error", "empty_after_normalize")
RULE_ORDER = ("R1", "R2", "R3", "R4", "R5", "MX", "TS", "BE", "PX", "SS", "TP", "R6")


def _rule(rules):
    return "+".join(r for r in RULE_ORDER if r in rules) or None


def _v(bucket, n_spans, extracted, layer, rule, detail, strict=None):
    """`strict` is v1's exact-layer verdict: bucket == exact_match, except where v2.5's TS refuses an exact match."""
    ok = bucket in ("exact_match", "equivalence_match")
    return {"bucket": bucket, "strict": bucket == "exact_match" if strict is None else strict, "lenient": ok,
            "n_boxed": n_spans, "extracted": extracted, "layer": layer, "rule": rule, "detail": detail}


def grade_one(completion, gold):
    """Grade a single completion against a gold answer string.

    `completion` is the model's text. `gold` is the answer key. They meet only here, and a failed extraction can never
    be replaced by anything derived from `gold`.
    """
    extracted, n_spans = extract_boxed(completion)
    if extracted is None:
        return _v("no_boxed", 0, None, None, None, "no_boxed")

    ne, ng = normalize(extracted), normalize(gold)
    if not ne or not ng:
        # Empty either side is never a match. Gold is asserted non-empty at load time.
        return _v("empty_after_normalize", n_spans, extracted, None, None, "empty")

    if ne == ng:                                    # exact normalised equality, never `in` (unchanged v1)
        if _ts_list_clash(extracted, gold, ne):     # v2.5 TS: "3,100" is not the list "3, 100"; strict stays v1's
            return _v("parsed_wrong", n_spans, extracted, "exact", "TS", "ts_vs_list", strict=True)
        return _v("exact_match", n_spans, extracted, "exact", None, "exact")

    try:
        with time_limit(TIMEOUT_S):                 # the string work is time-bound too (regexes on long spans)
            kind, a, b, rules, n_gold = _prepare(extracted, gold)
        if _ALARM["fired"]:
            raise GraderTimeout()
        with time_limit(TIMEOUT_S * max(1, n_gold)):    # v1's bound, once per gold item
            st, why = _decide(kind, a, b)
        if _ALARM["fired"]:                         # a swallowed alarm: whatever was computed after it is void
            raise GraderTimeout()
    except GraderTimeout:
        return _v("grader_timeout", n_spans, extracted, "timeout", "R6", "timeout")
    except GraderInternalError as ex:
        return _v("grader_error", n_spans, extracted, "error", "R6", f"error:{ex}")
    except Exception as ex:                         # anything else is the grader's fault: fail closed
        return _v("grader_error", n_spans, extracted, "error", "R6", f"error:{type(ex).__name__}")

    if st == MATCH:
        # strict=False deliberately: the strict layer is exact-match only (v1).
        return _v("equivalence_match", n_spans, extracted, "sympy", _rule(rules), why)
    if st == UNPARSEABLE:
        return _v("unparseable", n_spans, extracted, "sympy", "R5", why)
    return _v("parsed_wrong", n_spans, extracted, "sympy", _rule(rules), why)


# ---------------------------------------------------------------------------------------
# Command line: generation output -> counts file.            (added for this release)
# ---------------------------------------------------------------------------------------
def load_gold(path):
    """unique_id -> reference answer, from the public MATH-500 test split. Only grade_one ever sees it."""
    if path:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    else:
        from datasets import load_dataset
        rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    gold = {r["unique_id"]: r["answer"] for r in rows}
    assert len(gold) == len(rows), "unique_id collision"
    assert all(normalize(g) for g in gold.values()), "a reference answer normalises to empty"
    return gold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", required=True, help="generate_vllm.py output directory (T*.jsonl files)")
    ap.add_argument("--dataset", help="MATH-500 test split as JSONL (unique_id, answer); default: the HF hub")
    ap.add_argument("--out", required=True, help="counts JSONL to write")
    args = ap.parse_args()

    gold = load_gold(args.dataset)
    cache = {}                                    # (question id, extracted answer) -> verdict
    counts = {}
    tally = defaultdict(Counter)
    for path in sorted(glob.glob(os.path.join(args.gen_dir, "T*.jsonl"))):
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            qid, t = rec["id"], float(rec["temperature"])
            key_cell = (qid, t)
            assert key_cell not in counts, f"duplicate cell {key_cell}"
            n = c = c_strict = 0
            for comp in rec["completions"]:
                extracted, _ = extract_boxed(comp)
                key = (qid, extracted)
                if key not in cache:
                    cache[key] = grade_one(comp, gold[qid])
                v = cache[key]
                n += 1
                c += bool(v["lenient"])
                c_strict += bool(v["strict"])
                tally[t][v["bucket"]] += 1
            counts[key_cell] = {"id": qid, "temperature": t, "n": n, "c": c, "c_strict": c_strict}
    with open(args.out, "w") as f:
        for k in sorted(counts):
            f.write(json.dumps(counts[k]) + "\n")
    print(f"{len(counts)} cells written to {args.out}; buckets per temperature (completions):")
    for t in sorted(tally):
        tot = sum(tally[t].values())
        print(f"  T={t:g}: " + ", ".join(f"{b} {tally[t][b] / tot:.1%}" for b in BUCKETS if tally[t][b]))


if __name__ == "__main__":
    main()
