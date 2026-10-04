"""Build the unified WVS pilot table (mirrors build_dataset.py for GSS).

Reads the raw WVS Wave 7 inverted v6.0 CSV, recodes demographics into the
categorical scheme in config (agecat/sex/education/urbrural/country), drops WVS
missing codes (any negative raw value), and emits a long table of
(respondent, question) -> ordinal human answer, one row per pair.

Run with:  LLM_FAULTS_DATASET=WVS python build_dataset_wvs.py

Output: datasets/WVS/build/wvs_pilot_long.parquet (+ wvs_pilot_meta.json)
Same column contract as the GSS long table so all downstream scripts work unchanged.
"""
import json
import pandas as pd

import config as C

assert C.DATASET == "WVS", "run with LLM_FAULTS_DATASET=WVS"

# raw column -> our demographic name
RAW_DEMO = {"X003R": "agecat", "Q260": "sex", "Q275": "edu_raw",
            "H_URBRURAL": "urbrural", "B_COUNTRY": "country"}

# Q275 (highest educational level attained, ISCED 2011) -> our 4-level scheme.
# 0 early-childhood/none, 1 primary, 2 lower-sec  -> Primary or less (0) / Lower sec (1)
# 3 upper-sec, 4 post-sec non-tertiary, 5 short-cycle tertiary -> Upper/post-secondary (2)
# 6 bachelor, 7 master, 8 doctoral -> Tertiary (3)
EDU_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3}


def build():
    usecols = list(RAW_DEMO) + list(C.QUESTIONS) + ["D_INTERVIEW"]
    df = pd.read_csv(C.WVS_CSV, usecols=usecols, low_memory=False)
    n0 = len(df)

    # drop the duplicated interview id flagged by WorldValuesBench
    df = df[df["D_INTERVIEW"] != 858069901]

    df = df.rename(columns=RAW_DEMO)
    # coerce all to numeric; negatives / non-numeric -> NaN (WVS missing)
    for c in ["agecat", "sex", "edu_raw", "urbrural", "country"] + list(C.QUESTIONS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["education"] = df["edu_raw"].map(EDU_MAP)

    # keep respondents with all demographics present and valid
    for v in ["agecat", "sex", "education", "urbrural", "country"]:
        df = df[df[v].notna()]
    df = df[df["agecat"].between(1, 6)]
    df = df[df["sex"].isin([1, 2])]
    df = df[df["urbrural"].isin([1, 2])]
    df = df[df["country"].isin(C.DEMOGRAPHIC_CODES["country"].keys())]
    n2 = len(df)

    rows = []
    for qid, qmeta in C.QUESTIONS.items():
        codes = C.ANSWER_CODES[qid]
        valid = set(codes.keys())
        sub = df[df[qid].isin(valid)]
        for _, r in sub.iterrows():
            code = int(r[qid])
            rows.append({
                "dataset_id": "WVS",
                "respondent_id": f"{int(r['country'])}_{int(r['D_INTERVIEW'])}",
                "question_id": qid,
                "question_text": qmeta["text"],
                "topic": qmeta["topic"],
                "ordinal": qmeta["ordinal"],
                "agecat": int(r["agecat"]), "sex": int(r["sex"]),
                "education": int(r["education"]), "urbrural": int(r["urbrural"]),
                "country": int(r["country"]),
                "answer_code": code, "answer_text": codes[code],
                "n_options": len(codes),
            })
    long = pd.DataFrame(rows)

    C.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(C.BUILD_DIR / "wvs_pilot_long.parquet", index=False)

    meta = {
        "scope": {"dataset": "WVS Wave 7 inverted v6.0", "n_countries": int(long.country.nunique())},
        "rows_raw": n0, "rows_with_full_demographics": n2,
        "examples_total": len(long),
        "questions": {q: {"topic": C.QUESTIONS[q]["topic"],
                          "n_options": len(C.ANSWER_CODES[q]),
                          "n_examples": int((long.question_id == q).sum())}
                      for q in C.QUESTIONS},
        "demographic_vars": C.DEMOGRAPHIC_VARS, "subgroup_vars": C.SUBGROUP_VARS,
        "min_cell_humans": C.MIN_CELL_HUMANS,
    }
    (C.BUILD_DIR / "wvs_pilot_meta.json").write_text(json.dumps(meta, indent=2))
    return long, meta


if __name__ == "__main__":
    long, meta = build()
    print(f"raw rows          : {meta['rows_raw']:,}")
    print(f"full demographics : {meta['rows_with_full_demographics']:,}")
    print(f"countries         : {meta['scope']['n_countries']}")
    print(f"examples (long)   : {meta['examples_total']:,}")
    print("\nper-question examples:")
    for q, m in meta["questions"].items():
        print(f"  {q:6s} {m['n_examples']:7,d}  ({m['topic']}, {m['n_options']} opts)")
    print(f"\nwrote {C.BUILD_DIR/'wvs_pilot_long.parquet'}")
