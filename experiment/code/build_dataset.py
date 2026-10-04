"""Build the unified GSS pilot table.

Output: datasets/GSS/build/gss_pilot_long.parquet
  One row per (respondent, question) with columns:
    dataset_id, respondent_id, year, question_id, question_text, topic, ordinal,
    <demographic vars...>, answer_code, answer_text, n_options

Only respondents in [YEAR_MIN, YEAR_MAX] with a valid answer to the question and
non-missing values on ALL prompt demographics are kept (so every example can be
prompted and every subgroup assignment is well defined).

Also writes gss_pilot_meta.json: option lists per question, subgroup levels, counts.
"""
import json
import pandas as pd
import pyreadstat

import config as C


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """GSS columns mix int answer codes with string missing codes. Non-numeric -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def load_gss() -> pd.DataFrame:
    cols = list(dict.fromkeys(["year", "id"] + C.DEMOGRAPHIC_VARS + list(C.QUESTIONS)))
    df, _ = pyreadstat.read_dta(str(C.GSS_DTA), usecols=cols, encoding=C.STATA_ENCODING)
    for c in cols:
        df[c] = _coerce_numeric(df[c])
    return df


def build() -> pd.DataFrame:
    df = load_gss()
    n0 = len(df)
    df = df[(df["year"] >= C.YEAR_MIN) & (df["year"] <= C.YEAR_MAX)].copy()
    n1 = len(df)

    # keep only respondents with all demographics present and valid codes
    for v in C.DEMOGRAPHIC_VARS:
        df = df[df[v].notna()]
        if v in C.DEMOGRAPHIC_CODES:                       # categorical: restrict to known codes
            df = df[df[v].isin(C.DEMOGRAPHIC_CODES[v].keys())]
    df = df[df["age"].between(18, 89)]                     # sane adult age range
    n2 = len(df)

    rows = []
    for qid, qmeta in C.QUESTIONS.items():
        codes = C.ANSWER_CODES[qid]
        sub = df[df[qid].isin(codes.keys())].copy()
        for _, r in sub.iterrows():
            code = int(r[qid])
            rows.append({
                "dataset_id": "GSS",
                "respondent_id": f"{int(r['year'])}_{int(r['id'])}",
                "year": int(r["year"]),
                "question_id": qid,
                "question_text": qmeta["text"],
                "topic": qmeta["topic"],
                "ordinal": qmeta["ordinal"],
                **{v: int(r[v]) for v in C.DEMOGRAPHIC_VARS},
                "answer_code": code,
                "answer_text": codes[code],
                "n_options": len(codes),
            })
    long = pd.DataFrame(rows)

    C.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = C.BUILD_DIR / "gss_pilot_long.parquet"
    long.to_parquet(out, index=False)

    # metadata / provenance
    meta = {
        "scope": {"dataset": "GSS 7224 R3", "year_min": C.YEAR_MIN, "year_max": C.YEAR_MAX},
        "rows_raw": n0, "rows_in_years": n1, "rows_with_full_demographics": n2,
        "examples_total": len(long),
        "questions": {q: {"topic": C.QUESTIONS[q]["topic"],
                          "ordinal": C.QUESTIONS[q]["ordinal"],
                          "options": C.ANSWER_CODES[q],
                          "n_examples": int((long.question_id == q).sum())}
                      for q in C.QUESTIONS},
        "demographic_vars": C.DEMOGRAPHIC_VARS,
        "subgroup_vars": C.SUBGROUP_VARS,
        "min_cell_humans": C.MIN_CELL_HUMANS,
    }
    (C.BUILD_DIR / "gss_pilot_meta.json").write_text(json.dumps(meta, indent=2))
    return long, meta


if __name__ == "__main__":
    long, meta = build()
    print(f"raw rows            : {meta['rows_raw']:,}")
    print(f"in {C.YEAR_MIN}-{C.YEAR_MAX}        : {meta['rows_in_years']:,}")
    print(f"full demographics   : {meta['rows_with_full_demographics']:,}")
    print(f"examples (long)     : {meta['examples_total']:,}")
    print("\nper-question examples:")
    for q, m in meta["questions"].items():
        print(f"  {q:10s} {m['n_examples']:7,d}  ({m['topic']})")
    print(f"\nwrote {C.BUILD_DIR/'gss_pilot_long.parquet'}")
