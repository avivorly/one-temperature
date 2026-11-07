#!/usr/bin/env python3
"""MATH-500 grader of the paper (frozen version v2.5): completions -> counts.jsonl.

    python -m onetemp.grading.math500 --inputs runs/mistral_math500 --out counts_math500.jsonl --summary summary.json
    python -m onetemp.grading.math500 --try 'so the answer is $\\boxed{\\dfrac12}$' --gold '\\frac{1}{2}'

WHAT IT COMPUTES (Appendix C, "Datasets"). For every completion: the answer is the content of the LAST \\boxed{}
span of the completion, and a sample with no such span fails. A sample passes if its answer equals the reference
answer after normalization, or if an equivalence layer of generic rules and a time-bounded symbolic check with SymPy
accepts it. Per (question, setting) it writes n = samples and c = passing samples (instance-weighted), which is the
estimate c/n of the single-sample success probability P_q(tau) of Section 2. It also writes the optional field

    c_strict   samples whose answer passes the exact layer alone (normalized string equality), for reference.

The reference answers are the `answer` field of the public MATH-500 test split (HuggingFaceH4/MATH-500, which is
byte-identical to PRM800K's math_splits/test.jsonl). They are downloaded at run time and checked by sha256; none is
stored in this repository.

The grading code below is copied unchanged from the frozen grader that produced every MATH-500 number in the paper;
only comments and three docstrings were edited. It was frozen, with its hand-labelled test fixture, before the data
it graded existed, and it was never made more lenient after seeing results. Its design:

  * extract_boxed(completion) takes the completion ONLY. The problem record and the reference answer are never in
    its scope, so a failed extraction cannot be replaced by anything derived from the answer key.
  * The LAST \\boxed{} span counts, a rule fixed in advance; "pass if any span matches" would be in-sample oracle
    selection. The number of spans is recorded, never used.
  * No \\boxed{} is its own bucket (no_boxed) and fails. Empty never matches empty.
  * The exact layer is exact equality of normalized strings: never `in`, never a regex search.
  * Model output is never passed to sympify or eval directly, which would execute a string the model controls. Only
    a string of digits, letters, _, + - * / ( ) . and spaces reaches sympy's parse_expr, which evaluates it with a
    restricted namespace and empty __builtins__.
  * Every comparison is time-bound (2 s, SIGALRM; Unix main thread). A timeout or an internal error fails the
    sample (fail closed) and has its own bucket.
  * Only a real bool from sympy is believed (an unevaluated Eq is truthy).

The equivalence layer, in rule codes (recorded on each verdict as `rule`):
  R1 units       From both sides: every degree mark (^\\circ, ^{\\circ}, a Unicode degree sign, \\degree), and ONE
                 trailing unit block (\\text{..}, \\mbox{..} or \\mathrm{..} holding words, with a value before it)
                 together with a power directly after it (^2, ^{3}). The rest must still match.
  R2 prefix      A single-letter `v =` prefix is stripped once when it holds the side's only '='; on both sides the
                 letters must match. In a list, prefixes are stripped per element only when every element carries
                 the same letter.
  R3 lists       A reference that is a bare comma-separated list or a set \\{..\\} is a multiset: equal length,
                 matched element by element. Tuples, points, intervals, vectors and matrices stay ordered. A
                 comma-grouped number (12,345) is a number, not a list.
  R4 \\pm         An item `a \\pm b` (or \\mp) is the multiset {a+b, a-b}, on either side.
  R5 fail closed Any LaTeX command the translator does not handle (anything but brace-free \\frac, \\sqrt{}, \\pi,
                 \\times, \\div) refuses the sympy layer: bucket `unparseable`, a fail. A fixed whitelist of pure
                 layout commands (\\displaystyle, \\big, \\quad, ...) is stripped first.
  R6 buckets     A timeout -> grader_timeout, an internal error -> grader_error. Both fail, both are counted.
  MX mixed nums  A reference that is a mixed number (2\\frac{1}{3}) means addition. A candidate that writes its own
                 mixed number must be equal under both readings, the product a*b/c and the school reading a + b/c.
  TS thousands   A side that is ONE comma-grouped number (12,345 or 12{,}345) is that number without separators, only
                 when the other side is a single number. A comma followed by whitespace in the raw text separates a
                 list ("3, 100"), and a grouped number never meets a list, tuple, set or interval.
  BE exponents   ^{t} is ^t for one letter or digit t (layout only), except directly before a letter, digit or point.
  PX prefixes    Stripped once, like R2, when there is no other '=' or \\in: a subscripted or primed single letter
                 (z_1 =, x' =), an uppercase segment name (DE =), or `<v> \\in` before an interval or a set.
  SS singleton   A set \\{a\\} of one element against a scalar reference is a.
  TP tuples      A bare comma list against a parenthesised tuple reference of the same length, element by element
                 in order (an interval with \\infty is not a tuple).

Version history (each rule generic, never per question; each version's fixture written before its code):
  v2.0 R1-R6 (156 fixture cases); v2.1 more unit and degree forms (172); v2.2 the R2 whole-answer prefix and the R5
  layout whitelist (210); v2.3 a false pass removed: a mixed number must match under both readings (223); v2.4 a
  reference mixed number means addition (237); v2.5 TS, BE, PX, SS and TP, the final freeze (308 cases).

Output of grade_one(completion, gold), a dict:
  bucket   no_boxed | exact_match | equivalence_match | parsed_wrong | unparseable | grader_timeout | grader_error |
           empty_after_normalize
  lenient  the verdict (pass = exact_match or equivalence_match); strict = the exact layer alone
  n_boxed, extracted, layer, rule (the rule(s) that credited a pass, '+'-joined), detail (a reason code)

Reproducing the paper's verdicts needs sympy 1.14.0 and mpmath 1.3.0 with sympy's pure-Python ground types (the
script sets SYMPY_GROUND_TYPES=python and refuses other versions unless --any-sympy is given).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter

from . import completions_io as io_  # noqa: E402

# Pure-Python ground types, as in the paper's runs; must be set before sympy is first imported.
os.environ["SYMPY_GROUND_TYPES"] = "python"

# The block below is the grader the paper's verdicts came from, byte for byte, so its own comments are kept as
# they were: the test that pins it by sha256 is tests/test_grading_math500.py in this repository.
# =======================================================================================
# THE FROZEN GRADER (v2.5). Code unchanged; tests/test_math500.py pins this block by the sha256 of its text.
# =======================================================================================
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

    v2.3 (a false pass found in an audit): when either side holds a mixed number, the two sides must be equal under BOTH readings --
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
# R3: a comma-grouped number is a number, not a list. v2.5 TS: this pattern, on a side's form ({,} read as a
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


# =======================================================================================
# END OF THE FROZEN GRADER. Everything below drives it (added for this release).
# =======================================================================================

# The public MATH-500 test split. HuggingFaceH4/MATH-500 at this revision is byte-identical to openai/prm800k's
# prm800k/math_splits/test.jsonl, whose git-LFS pointer records the same sha256 and 446,564 bytes.
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_URL = f"https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/{MATH500_REVISION}/test.jsonl"
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
SYMPY_MPMATH = ("1.14.0", "1.3.0")      # the versions behind every verdict in the paper
RECHECK_EVERY = 97                      # every 97th cache hit is graded again on its own, as in the paper's runs
BUCKET_CODE = {"no_boxed": "n", "exact_match": "e", "equivalence_match": "q", "parsed_wrong": "w", "unparseable": "u",
               "grader_timeout": "t", "grader_error": "x", "empty_after_normalize": "z"}


def load_gold(path):
    """unique_id -> reference answer string, from the MATH-500 test split (one JSON object per line).

    Only grade_one ever sees a reference answer. Every reference must normalize to a non-empty string, so that an
    empty extraction can never equal an empty reference."""
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    gold = {r["unique_id"]: r["answer"] for r in rows}
    if len(gold) != len(rows):
        raise SystemExit(f"{path}: duplicate unique_id")
    if not all(isinstance(g, str) and g.strip() and normalize(g) for g in gold.values()):
        raise SystemExit(f"{path}: a reference answer is empty after normalization")
    return gold


def check_sympy(allow_other):
    """The grader's environment, checked rather than assumed: sympy/mpmath versions and pure-Python ground types."""
    import mpmath
    import sympy
    from sympy.external.gmpy import GROUND_TYPES
    env = {"sympy": sympy.__version__, "mpmath": mpmath.__version__, "sympy_ground_types": GROUND_TYPES,
           "python": sys.version.split()[0]}
    problems = []
    if (sympy.__version__, mpmath.__version__) != SYMPY_MPMATH:
        problems.append(f"sympy/mpmath {sympy.__version__}/{mpmath.__version__}, the paper used "
                        f"{SYMPY_MPMATH[0]}/{SYMPY_MPMATH[1]}")
    if GROUND_TYPES != "python":
        problems.append(f"sympy ground types {GROUND_TYPES!r}, the paper used 'python'")
    if problems and not allow_other:
        raise SystemExit("; ".join(problems) + ". Install the grading extra (pip install -e \".[grading]\"), or pass "
                         "--any-sympy to grade anyway (verdicts may then differ from the paper's).")
    for p in problems:
        print(f"WARNING: {p}", file=sys.stderr)
    return env


def _fail_closed(text, gold):
    """grade_one never raises; if it ever did, the sample would be a grader_error (a fail), never a pass."""
    try:
        return grade_one(text, gold)
    except KeyboardInterrupt:
        raise
    except BaseException:  # noqa: BLE001
        return {"bucket": "grader_error", "strict": False, "lenient": False, "rule": "R6"}


def grade_texts(gold, texts, recheck_every=RECHECK_EVERY):
    """Per-sample verdicts for the completions of ONE cell, instance-weighted.

    grade_one's verdict depends only on the last \\boxed{} span and the reference, so a distinct span is graded once
    per cell and its verdict is reused for every sample with that span. As a check, a new distinct text whose span is
    already graded is graded again on its own whenever the number of distinct texts seen so far is a multiple of
    `recheck_every` (identical repeats of a text are never re-graded). A disagreement that involves no timeout is
    counted in `cache_mismatches`, and the caller then fails the run. This is the caching of the paper's runs, cell
    by cell. Returns per-sample bit strings and counts."""
    vt, vindex, span_cache, text_cache = [], {}, {}, {}
    checks = mismatches = 0
    idx, nbox = [], []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()
        hit = text_cache.get(h)
        if hit is None:
            ex, nb = extract_boxed(text)
            v = span_cache.get(ex)
            if v is None:
                r = _fail_closed(text, gold)
                v = (r["bucket"], bool(r["strict"]), bool(r["lenient"]), r.get("rule"))
                span_cache[ex] = v
            elif recheck_every and len(text_cache) % recheck_every == 0:
                r = _fail_closed(text, gold)
                checks += 1
                if (r["bucket"], bool(r["lenient"]), r.get("rule")) != (v[0], v[2], v[3]) \
                        and "grader_timeout" not in (r["bucket"], v[0]):
                    mismatches += 1
            if v not in vindex:
                vindex[v] = len(vt)
                vt.append(v)
            hit = text_cache[h] = (vindex[v], nb)
        idx.append(hit[0])
        nbox.append(hit[1])
    return {"pass": "".join("1" if vt[i][2] else "0" for i in idx),
            "strict": "".join("1" if vt[i][1] else "0" for i in idx),
            "bucket": "".join(BUCKET_CODE[vt[i][0]] for i in idx),
            "multi_box": sum(1 for nb in nbox if nb > 1),
            "distinct_texts": len(text_cache), "distinct_spans": len(span_cache),
            "cache_checks": checks, "cache_mismatches": mismatches}


# --------------------------------------------------------------------------- worker processes
_WORKER = {}


def _init_worker(rlimit_gb):
    """Each grading process caps its own address space (3 GB in the paper's runs): a pathological expression then
    raises MemoryError inside sympy, which the grader turns into grader_error, instead of exhausting the machine."""
    if rlimit_gb:
        try:
            import resource
            lim = int(rlimit_gb * (1 << 30))
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (lim, hard if hard != resource.RLIM_INFINITY else lim))
        except (ImportError, ValueError, OSError):
            pass
    _WORKER["ready"] = True


def _grade_job(job):
    """job = (gold answer, completions, finish_reasons) -> per-cell result. Runs in a worker's main thread, where
    SIGALRM works."""
    gold, texts, fins = job
    res = grade_texts(gold, texts)
    try:
        from sympy.core.cache import clear_cache
        clear_cache()                                # once per cell, as in the paper's runs
    except Exception:  # noqa: BLE001
        pass
    trunc = "".join("1" if f == "length" else "0" for f in fins) if fins is not None else None
    return res, trunc


def grade_inputs(paths, gold, jobs=1, model=None, rlimit_gb=3.0):
    """Grade every cell under `paths` -> (CountsTable, Summary, totals). Gold answers stay in this process and in
    the grading workers; they are never written out."""
    import concurrent.futures as cf
    counts = io_.CountsTable(extra_fields=("c_strict",))
    summary = io_.Summary()
    totals = Counter()
    cells = io_.iter_cells(paths, "math500", model=model)
    with cf.ProcessPoolExecutor(max_workers=max(1, jobs), initializer=_init_worker, initargs=(rlimit_gb,)) as ex:
        pending = {}

        def drain(block):
            done, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED if not block else cf.ALL_COMPLETED)
            for fut in done:
                cell = pending.pop(fut)
                res, trunc = fut.result()
                if res["cache_mismatches"]:
                    raise SystemExit(f"{cell['source']}: {res['cache_mismatches']} cache re-checks disagreed with "
                                     f"the cached verdict; the run is not trusted")
                n = len(res["pass"])
                c, cs = res["pass"].count("1"), res["strict"].count("1")
                counts.add(cell, n, c, {"c_strict": cs}, texts_hash=cell["_hash"])
                code_bucket = {v: k for k, v in BUCKET_CODE.items()}
                buckets = Counter(code_bucket[ch] for ch in res["bucket"])
                nt = trunc.count("1") if trunc else 0
                ntp = sum(1 for p, t in zip(res["pass"], trunc) if p == "1" and t == "1") if trunc else 0
                summary.add(cell, buckets, nt, ntp)
                totals.update({"cells": 1, "samples": n, "passes": c, "multi_boxed": res["multi_box"],
                               "distinct_spans": res["distinct_spans"], "cache_checks": res["cache_checks"]})

        for cell in cells:
            if cell["id"] not in gold:
                raise SystemExit(f"{cell['source']}: {cell['id']!r} is not a MATH-500 unique_id")
            cell["_hash"] = io_.texts_sha256(cell["completions"])
            fut = ex.submit(_grade_job, (gold[cell["id"]], cell["completions"], cell["finish_reasons"]))
            cell.pop("completions")
            pending[fut] = cell
            if len(pending) >= 2 * max(1, jobs):
                drain(False)
        while pending:
            drain(True)
    return counts, summary, totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", help="completions files or directories (*.jsonl, *.jsonl.gz)")
    ap.add_argument("--out", help="counts.jsonl to write")
    ap.add_argument("--summary", help="optional JSON: buckets and truncation per (model, setting), counts only")
    ap.add_argument("--dataset", help="local copy of the MATH-500 test.jsonl (default: download and cache it)")
    ap.add_argument("--cache-dir", default=io_.default_cache_dir(), help="where downloaded data is kept")
    ap.add_argument("--no-verify-data", action="store_true", help="skip the sha256 check of the dataset")
    ap.add_argument("--model", help="model id for lines that do not name one")
    ap.add_argument("--jobs", type=int, default=1, help="grading processes (default 1)")
    ap.add_argument("--rlimit-gb", type=float, default=3.0, help="address-space cap per grading process, GB "
                    "(the paper's three-model grids: 3, its GPT-OSS grading: 6; 0 = none)")
    ap.add_argument("--any-sympy", action="store_true", help="run even if sympy/mpmath are not the paper's versions")
    ap.add_argument("--force", action="store_true", help="overwrite --out / --summary")
    ap.add_argument("--try", dest="try_text", metavar="COMPLETION", help="grade one completion against --gold and "
                    "print the verdict")
    ap.add_argument("--gold", help="reference answer for --try")
    a = ap.parse_args(argv)

    env = check_sympy(a.any_sympy)
    if a.try_text is not None:
        if a.gold is None:
            ap.error("--try needs --gold")
        print(json.dumps(grade_one(a.try_text, a.gold), ensure_ascii=False))
        return 0
    if not a.inputs or not a.out:
        ap.error("--inputs and --out are required")
    out_path, summary_path = io_.check_outputs([a.out, a.summary], force=a.force)     # before any work
    path = io_.fetch_pinned(MATH500_URL, MATH500_SHA256, a.cache_dir, "math500_test.jsonl", a.dataset,
                            check=not a.no_verify_data)
    gold = load_gold(path)
    counts, summary, totals = grade_inputs(a.inputs, gold, jobs=a.jobs, model=a.model, rlimit_gb=a.rlimit_gb)
    rows = counts.sorted_rows()
    io_.write_jsonl_atomic(out_path, rows, force=a.force)
    for w in counts.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if summary_path:
        io_.write_json_atomic(summary_path, {
            "benchmark": "math500", "grader_version": GRADER_VERSION, "environment": env,
            "dataset": {"url": MATH500_URL, "sha256": io_.sha256_file(path), "verified": not a.no_verify_data},
            "totals": dict(totals), "by_setting": summary.as_json()}, force=a.force)
    print(f"{len(rows)} cells, {totals['samples']} samples, {totals['passes']} passing -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
