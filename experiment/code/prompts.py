"""Prompt construction for the pilot (dataset-agnostic).

Prompt styles:
  A  minimal-demographic  -> return a single answer key
  B  natural-language persona -> single answer key (RQ4 robustness only)
  C  distribution-aware   -> return a JSON probability distribution over keys

The "answer key" is a letter (A, B, C ...) when options have short text labels
(GSS: favor/oppose, etc.), or the scale number itself when options are numeric
points on an ordinal scale (WVS: 1..10). Numeric keys avoid the 8-letter limit and
read naturally for long scales. In both cases build_prompt returns a mapping
{key -> answer_code} that the parser uses to recover the code.

Styles A and C render the same demographics and differ only in the response contract,
which lets us test individual-level fidelity (A) and calibration (C) from one sample.
"""
import config as C

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _demographics_block(row) -> str:
    """Render the respondent's demographics as human-readable lines."""
    lines = []
    for v in C.DEMOGRAPHIC_VARS:
        code = int(row[v])
        if v == "age":                       # GSS continuous age
            val = "89 or older" if code >= 89 else str(code)
        else:
            val = C.DEMOGRAPHIC_CODES[v].get(code, str(code))
        lines.append(f"- {C.DEMOGRAPHIC_QUESTION_TEXT[v]}: {val}")
    return "\n".join(lines)


def _options_block(qid, reverse=False):
    """Return (rendered options text, {key: answer_code}) preserving code order.

    Numeric key mode: when every option label equals its own code (ordinal scale),
    the answer key is the number itself. Otherwise use letters.

    reverse=True presents the options in reversed display order. For letter keys
    this makes letter A point to a *different* underlying answer, so a positional
    bias (anchoring on the first letter) shows up as a shift in the recovered
    answer distribution; the key->code mapping still recovers the true code. This
    is the option-order robustness perturbation (RQ4).
    """
    codes = list(C.ANSWER_CODES[qid].keys())
    labels = C.ANSWER_CODES[qid]
    numeric_keys = all(str(labels[c]) == str(c) for c in codes)
    if reverse:
        codes = list(reversed(codes))

    mapping, lines = {}, []
    for i, code in enumerate(codes):
        key = str(code) if numeric_keys else LETTERS[i]
        mapping[key] = code
        if numeric_keys:
            lines.append(f"{key}")
        else:
            lines.append(f"[{key}] {labels[code]}")
    # for numeric scales, render compactly on one line
    opts = "  ".join(lines) if numeric_keys else "\n".join(lines)
    return opts, mapping, numeric_keys


def _persona_sentence(row) -> str:
    """Render demographics as a natural-language persona sentence (style B)."""
    parts = []
    for v in C.DEMOGRAPHIC_VARS:
        code = int(row[v])
        if v == "age":
            val = "89 or older" if code >= 89 else str(code)
        else:
            val = C.DEMOGRAPHIC_CODES[v].get(code, str(code))
        parts.append(f"{C.DEMOGRAPHIC_QUESTION_TEXT[v].lower()} {val}")
    return "a person with " + ", ".join(parts)


def build_prompt(row, style: str, reverse: bool = False):
    """Return (prompt_text, key->code mapping) for a pilot row and prompt style.

    reverse=True permutes the option display order (option-order robustness, RQ4).
    style 'B' is a rich natural-language persona framing of the same demographics.
    """
    qid = row["question_id"]
    demo = _demographics_block(row)
    opts, mapping, numeric_keys = _options_block(qid, reverse=reverse)
    qtext = row["question_text"]
    noun = C.RESPONDENT_NOUN
    key_word = "number" if numeric_keys else "option letter"
    example = "3" if numeric_keys else "A"

    if style == "A":
        prompt = (
            f"You are simulating a single {noun} with the following demographic profile:\n\n"
            f"{demo}\n\n"
            "Based only on this profile, predict how this specific person would answer "
            "the following survey question.\n\n"
            f"Question:\n{qtext}\n\n"
            f"Answer options:\n{opts}\n\n"
            f"Respond with only the single {key_word} (e.g. {example}). No explanation."
        )
    elif style == "B":
        persona = _persona_sentence(row)
        prompt = (
            f"Imagine {persona}. This is {('an' if noun[0] in 'aeiou' else 'a')} {noun}.\n\n"
            "Thinking about how this particular person sees the world, predict how they "
            "would answer the following survey question.\n\n"
            f"Question:\n{qtext}\n\n"
            f"Answer options:\n{opts}\n\n"
            f"Respond with only the single {key_word} (e.g. {example}). No explanation."
        )
    elif style == "C":
        keys = ", ".join(f'"{k}"' for k in mapping)
        prompt = (
            f"Consider a {noun} with the following demographic profile:\n\n"
            f"{demo}\n\n"
            "Estimate the probability that this respondent would choose each answer "
            "option for the question below. Probabilities must sum to 1.\n\n"
            f"Question:\n{qtext}\n\n"
            f"Answer options:\n{opts}\n\n"
            f"Return only valid JSON mapping each answer {key_word} to a probability, "
            f"using exactly these keys: {{{keys}}}. No other text."
        )
    else:
        raise ValueError(f"unknown style {style!r}")
    return prompt, mapping
