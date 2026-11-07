#!/usr/bin/env python3
"""Prompt builders: the exact prompt of every (benchmark, model) of the paper, as token ids (Appendix C "Datasets").

Each builder takes ONE question's public text (never its answer) and a Hugging Face tokenizer, and returns
    {"prompt_token_ids": [...], "prompt_text": str}
where prompt_text is the decoded prompt with special tokens shown. The sampler sends prompt_token_ids to vLLM
unchanged, so every byte of every prompt is fixed here, off the GPU, and can be hashed and checked.

Styles (`STYLES`), by benchmark:

MATH-500. Every model gets the instruction
    "Solve the following math problem efficiently and clearly. Reason step by step, and put your final answer
     within \\boxed{}."
as the system message and the problem as the user message, in its own chat template. Qwen3-8B gets the instruction
followed by one sentence that states its 2,048-token limit.
  math500_mistral           Mistral template; v0.3 folds the system text into the user turn:
                            "<s>[INST] {instruction}\\n\\n{problem}[/INST]"
  math500_qwen3             Qwen3 template, add_generation_prompt, thinking disabled (empty think block)
  math500_qwen3_8b_concise  the same with the limit sentence appended to the instruction
  math500_harmony           GPT-OSS Harmony template (its system message carries "Reasoning: medium"; our
                            instruction becomes the developer message), date line pinned to 2026-10-10, and the
                            response primed to open in the final channel: "...<|start|>assistant<|channel|>final
                            <|message|>"

Hi-ToM. The user message is the dataset's own `prompt` field, unchanged (it lists the 15 options A-O); no system
message. Mistral gets an instruction that states its 500-token limit appended to it.
  hitom_mistral_cap500      "<s>[INST] {prompt}{CAP500}[/INST]"
  hitom_qwen3               Qwen3 template, add_generation_prompt, thinking disabled
  hitom_harmony             Harmony with reasoning effort medium, date pinned to 2026-10-10, no developer message,
                            response primed with <|channel|>final<|message|>

LiveCodeBench. A body is built from the problem statement and, for call-based (LeetCode) problems, the starter code;
the system text asks for a short solution under 500 tokens; the response is primed to open a ```python block.
  lcb_raw                   Qwen3-0.6B: no chat template; system text, blank line, body, then "```python\\n"
  lcb_mistral               Mistral template ids + ids of "\\n```python\\n"
  lcb_qwen3                 Qwen3 template (thinking disabled) ids + ids of "```python\\n"
  lcb_harmony               Harmony template ids + ids of "<|channel|>final<|message|>```python\\n"; the date
                            line is the render day unless pinned (the runs rendered it on the GPU machine)

The token-id concatenations of the LiveCodeBench styles are done exactly as the paper's generators did them
(template ids, then the separately encoded priming), because encoding the joined text can merge tokens across the
boundary.
"""
import hashlib
import json
import re

# ---------------------------------------------------------------------------------------------------------------
# Texts (verbatim from the paper's runs)
# ---------------------------------------------------------------------------------------------------------------
MATH500_INSTRUCTION = ("Solve the following math problem efficiently and clearly. Reason step by step, and put your "
                       "final answer within \\boxed{}.")
# Qwen3-8B (2,048-token limit): the instruction, one space, then this sentence
MATH500_QWEN3_8B_SENTENCE = ("Your answer can be at most 2,048 tokens long, so keep it very concise and end it with the "
                             "final answer in \\boxed{}.")
MATH500_QWEN3_8B_INSTRUCTION = MATH500_INSTRUCTION + " " + MATH500_QWEN3_8B_SENTENCE

# Mistral on Hi-ToM: appended to the dataset prompt (which ends in "\n"). Two em dashes (U+2014).
HITOM_CAP500_SUFFIX = ("\n\nAnswer length limit: your reply is capped at 500 tokens (roughly 350 words) and will be cut "
                       "off at that point. Give the answer FIRST — the choice letter and the container name "
                       "— and only then explain, briefly. If you run out of room while explaining, that is fine; "
                       "the answer has already been given.")

LCB_SYSTEM = "You are an expert Python programmer. Write a short, correct solution (under 500 tokens)."
LCB_STDIN_LINE = "Read from stdin, write to stdout. Enclose code in ```python``` delimiters.\n\n"

HARMONY_PIN_DATE = "2026-10-10"          # the date line of every MATH-500 and Hi-ToM GPT-OSS prompt
HARMONY_FINAL_PRIME = "<|channel|>final<|message|>"
# What the gpt-oss-20b template emits around our texts (reasoning effort medium), written out independently of it
HARMONY_SYSTEM_BLOCK = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
                        "Knowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: medium\n\n# Valid channels: "
                        "analysis, commentary, final. Channel must be included for every message.<|end|>")
HARMONY_DEVELOPER_BLOCK = "<|start|>developer<|message|># Instructions\n\n{text}\n\n<|end|>"
HARMONY_USER_BLOCK = "<|start|>user<|message|>{text}<|end|>"
QWEN3_NOTHINK_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_DATE_RE = re.compile(r"Current date: (\d{4}-\d{2}-\d{2})\n")

# Expected sha256 of the instruction texts (as recorded by the runs), checked at import
_SHA = {MATH500_INSTRUCTION: "468adbdf632b7dafbcec2d24bcc0eda6638b0228585877e4738b6135242cf58f",
        MATH500_QWEN3_8B_INSTRUCTION: "f2ac449e92465c434d26bcd967e804c90178fd7d1563fe0c84bbe13ae027ca51"}


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ids_sha256(ids):
    """sha256 of json.dumps(list of ints), the definition the runs used for prompt-id hashes."""
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


for _t, _h in _SHA.items():
    assert sha256_text(_t) == _h, "instruction text differs from the paper's"
assert len(HITOM_CAP500_SUFFIX) == 301


# ---------------------------------------------------------------------------------------------------------------
# Tokenizer helpers (work with transformers 4.x and 5.x, where apply_chat_template(tokenize=True) may return a dict)
# ---------------------------------------------------------------------------------------------------------------
def _as_ids(x):
    if hasattr(x, "keys"):
        x = x["input_ids"]
    x = list(x)
    if x and isinstance(x[0], list):          # a batch of one
        assert len(x) == 1
        x = x[0]
    return [int(i) for i in x]


def _template_ids(tok, msgs, **kw):
    return _as_ids(tok.apply_chat_template(msgs, tokenize=True, **kw))


def _template_text(tok, msgs, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, **kw)


def _encode(tok, text):
    return _as_ids(tok.encode(text, add_special_tokens=False))


def _decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def _pin_harmony_date(text, date):
    found = _DATE_RE.findall(text)
    if len(found) != 1:
        raise ValueError(f"expected one 'Current date: YYYY-MM-DD' line in the Harmony prompt, found {found}")
    if date is None:
        return text
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"--harmony-date must be YYYY-MM-DD, got {date!r}")
    return _DATE_RE.sub(f"Current date: {date}\n", text, count=1)


def _no_markup(s, marks, where):
    for m in marks:
        if m in s:
            raise ValueError(f"{where}: the text contains {m!r}, which the tokenizer would read as a control token")


# ---------------------------------------------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------------------------------------------
def math500_messages(problem, instruction=MATH500_INSTRUCTION):
    if "\\boxed{" in problem:
        raise ValueError("problem text carries a \\boxed{} span")
    return [{"role": "system", "content": instruction}, {"role": "user", "content": problem}]


def build_math500_mistral(q, tok, **_):
    msgs = math500_messages(q["problem"])
    _no_markup(q["problem"], ("[INST]", "[/INST]", "<s>", "</s>"), q["id"])
    ids = _template_ids(tok, msgs)
    text = _template_text(tok, msgs)
    expect = "<s>[INST] " + MATH500_INSTRUCTION + "\n\n" + q["problem"] + "[/INST]"
    if text != expect:
        raise ValueError(f"{q['id']}: the Mistral template renders differently from the paper's v0.3 template")
    return {"prompt_token_ids": ids, "prompt_text": text}


def _math500_qwen3(q, tok, instruction):
    msgs = math500_messages(q["problem"], instruction)
    _no_markup(q["problem"], ("<|im_", "<think>", "</think>", "<|endoftext|>"), q["id"])
    ids = _template_ids(tok, msgs, add_generation_prompt=True, enable_thinking=False)
    text = _template_text(tok, msgs, add_generation_prompt=True, enable_thinking=False)
    expect = ("<|im_start|>system\n" + instruction + "<|im_end|>\n<|im_start|>user\n" + q["problem"] +
              "<|im_end|>\n" + QWEN3_NOTHINK_TAIL)
    if text != expect:
        raise ValueError(f"{q['id']}: the Qwen3 template renders differently from the paper's")
    return {"prompt_token_ids": ids, "prompt_text": text}


def build_math500_qwen3(q, tok, **_):
    return _math500_qwen3(q, tok, MATH500_INSTRUCTION)


def build_math500_qwen3_8b_concise(q, tok, **_):
    return _math500_qwen3(q, tok, MATH500_QWEN3_8B_INSTRUCTION)


def _harmony_date_of(text):
    return _DATE_RE.findall(text)[0]


def build_math500_harmony(q, tok, harmony_date=HARMONY_PIN_DATE, **_):
    msgs = math500_messages(q["problem"])
    _no_markup(q["problem"], ("<|",), q["id"])
    text = _template_text(tok, msgs, add_generation_prompt=True)
    text = _pin_harmony_date(text, harmony_date) + HARMONY_FINAL_PRIME
    expect = (HARMONY_SYSTEM_BLOCK.format(date=_harmony_date_of(text)) +
              HARMONY_DEVELOPER_BLOCK.format(text=MATH500_INSTRUCTION) + HARMONY_USER_BLOCK.format(text=q["problem"]) +
              "<|start|>assistant" + HARMONY_FINAL_PRIME)
    if text != expect:
        raise ValueError(f"{q['id']}: the gpt-oss template renders differently from the paper's Harmony layout")
    ids = _encode(tok, text)
    return {"prompt_token_ids": ids, "prompt_text": text}


# ---------------------------------------------------------------------------------------------------------------
# Hi-ToM
# ---------------------------------------------------------------------------------------------------------------
def build_hitom_mistral_cap500(q, tok, **_):
    content = q["prompt"] + HITOM_CAP500_SUFFIX
    msgs = [{"role": "user", "content": content}]
    ids = _template_ids(tok, msgs)
    text = _template_text(tok, msgs)
    if text != "<s>[INST] " + content + "[/INST]":
        raise ValueError(f"{q['id']}: the Mistral template renders differently from the paper's v0.3 template")
    return {"prompt_token_ids": ids, "prompt_text": text}


def build_hitom_qwen3(q, tok, **_):
    msgs = [{"role": "user", "content": q["prompt"]}]
    text = _template_text(tok, msgs, add_generation_prompt=True, enable_thinking=False)
    if text != "<|im_start|>user\n" + q["prompt"] + "<|im_end|>\n" + QWEN3_NOTHINK_TAIL:
        raise ValueError(f"{q['id']}: the Qwen3 template renders differently from the paper's")
    ids = _encode(tok, text)             # the runs tokenized this text with add_special_tokens=False
    return {"prompt_token_ids": ids, "prompt_text": text}


def build_hitom_harmony(q, tok, harmony_date=HARMONY_PIN_DATE, **_):
    msgs = [{"role": "user", "content": q["prompt"]}]
    text = _template_text(tok, msgs, add_generation_prompt=True, reasoning_effort="medium")
    text = _pin_harmony_date(text, harmony_date) + HARMONY_FINAL_PRIME
    expect = (HARMONY_SYSTEM_BLOCK.format(date=_harmony_date_of(text)) + HARMONY_USER_BLOCK.format(text=q["prompt"]) +
              "<|start|>assistant" + HARMONY_FINAL_PRIME)
    if text != expect:
        raise ValueError(f"{q['id']}: the gpt-oss template renders differently from the paper's Harmony layout")
    ids = _encode(tok, text)
    return {"prompt_token_ids": ids, "prompt_text": text}


# ---------------------------------------------------------------------------------------------------------------
# LiveCodeBench
# ---------------------------------------------------------------------------------------------------------------
def lcb_body(question_content, starter_code):
    """The user-message body of every LiveCodeBench prompt (all four models)."""
    body = question_content + "\n\n"
    if starter_code:
        body += "Complete this code:\n"
        body += "```python\n"
        body += starter_code
        body += "\n```\n\n"
    else:
        body += LCB_STDIN_LINE
    return body


def lcb_messages(q):
    return [{"role": "system", "content": LCB_SYSTEM},
            {"role": "user", "content": lcb_body(q["question_content"], q["starter_code"])}]


def build_lcb_raw(q, tok, **_):
    """Qwen3-0.6B: a plain completion prompt, which vLLM tokenized itself (the Qwen tokenizer adds no BOS)."""
    text = LCB_SYSTEM + "\n\n" + lcb_body(q["question_content"], q["starter_code"]) + "```python\n"
    ids = _as_ids(tok(text)["input_ids"])
    if ids != _encode(tok, text):
        raise ValueError("this tokenizer adds special tokens to a plain prompt; the paper's did not")
    return {"prompt_token_ids": ids, "prompt_text": text}


def build_lcb_mistral(q, tok, **_):
    msgs = lcb_messages(q)
    ids = _template_ids(tok, msgs) + _encode(tok, "\n```python\n")
    return {"prompt_token_ids": ids, "prompt_text": _decode(tok, ids)}


def build_lcb_qwen3(q, tok, **_):
    msgs = lcb_messages(q)
    ids = _template_ids(tok, msgs, add_generation_prompt=True, enable_thinking=False) + _encode(tok, "```python\n")
    return {"prompt_token_ids": ids, "prompt_text": _decode(tok, ids)}


def build_lcb_harmony(q, tok, harmony_date=None, **_):
    """The runs used apply_chat_template(tokenize=True) and the render day's date. With a pinned date the text is
    rendered, its date line replaced, and encoded (the same ids up to the date characters)."""
    msgs = lcb_messages(q)
    prime = HARMONY_FINAL_PRIME + "```python\n"
    if harmony_date is None:
        ids = _template_ids(tok, msgs, add_generation_prompt=True) + _encode(tok, prime)
    else:
        text = _pin_harmony_date(_template_text(tok, msgs, add_generation_prompt=True), harmony_date)
        ids = _encode(tok, text) + _encode(tok, prime)
    return {"prompt_token_ids": ids, "prompt_text": _decode(tok, ids)}


STYLES = {
    "math500_mistral": ("math500", build_math500_mistral),
    "math500_qwen3": ("math500", build_math500_qwen3),
    "math500_qwen3_8b_concise": ("math500", build_math500_qwen3_8b_concise),
    "math500_harmony": ("math500", build_math500_harmony),
    "hitom_mistral_cap500": ("hitom", build_hitom_mistral_cap500),
    "hitom_qwen3": ("hitom", build_hitom_qwen3),
    "hitom_harmony": ("hitom", build_hitom_harmony),
    "lcb_raw": ("livecodebench", build_lcb_raw),
    "lcb_mistral": ("livecodebench", build_lcb_mistral),
    "lcb_qwen3": ("livecodebench", build_lcb_qwen3),
    "lcb_harmony": ("livecodebench", build_lcb_harmony),
}
# The Harmony styles whose date line the paper pinned by default (LiveCodeBench: render day)
DEFAULT_HARMONY_DATE = {"math500_harmony": HARMONY_PIN_DATE, "hitom_harmony": HARMONY_PIN_DATE, "lcb_harmony": None}


def build(style, q, tok, harmony_date="default"):
    """One prompt record for question q in this style. q holds only public question text (see public_data.py)."""
    if style not in STYLES:
        raise SystemExit(f"unknown prompt style {style!r}; known: {sorted(STYLES)}")
    if harmony_date == "default":
        harmony_date = DEFAULT_HARMONY_DATE.get(style)
    out = STYLES[style][1](q, tok, harmony_date=harmony_date)
    ids = out["prompt_token_ids"]
    if not ids or not all(isinstance(i, int) for i in ids):
        raise ValueError(f"{q['id']}: empty or non-integer prompt ids")
    out["n_prompt_tokens"] = len(ids)
    out["prompt_ids_sha256"] = ids_sha256(ids)
    out["prompt_text_sha256"] = sha256_text(out["prompt_text"])
    return out
