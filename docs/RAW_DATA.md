# Obtaining the raw survey data

**You almost certainly do not need this.** Every number, table, and figure in the
paper reproduces from the cached model outputs already in this repository, with no
raw data and no model calls. See [REPRODUCE.md](REPRODUCE.md).

You need the raw files only to:

1. Rebuild the derived tables from scratch (Stage 1),
2. Run `analyze_weights.py` — the one analysis script that re-reads the raw files,
   because survey weights (`wtssps`, `W_WEIGHT`) were never copied into the build
   tables. Its published output is already included, or
3. **Run `analyze_rq1_robustness.py` on WVS** (Table 2 — RQ1 robustness). It needs the
   full `wvs_pilot_long.parquet`, which we do not redistribute because the WVSA
   usage agreement restricts it. Rebuilding it after your own WVS download takes
   ~10 minutes. Its published output is also already included.

> **On the WVS agreement.** WVS microdata is not included. Each user must accept
> the WVSA terms themselves. What this
> repository contains from WVS is a 1,985-row schema demonstration sample
> (`wvs_pilot_long_SAMPLE.parquet`, 126 respondents = 0.14% of the sample), the
> 1,458-row evaluation set needed to verify the paper, and aggregate distributions.
> GSS is different: NORC distributes it publicly with no registration and no
> agreement, so the GSS derived table ships in full.

Both files are free. **Neither has a stable direct-download URL** — GSS serves from
a CDN path that changes each release, and WVS requires accepting a usage agreement
first. Both steps below are manual browser downloads. Exact filenames matter:
`config.py` looks for these paths literally.

| File | Must land at | Size |
|---|---|---|
| `gss7224_r3.dta` | `experiment/datasets/GSS/gss7224_r3.dta` | 571 MB |
| `WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv` | `experiment/datasets/WVS_raw/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv` | 182 MB |

Both paths are in `.gitignore`. Do not commit them.

---

## GSS — General Social Survey cumulative file, 1972–2024 (Release 3)

1. Go to <https://gss.norc.org/us/en/gss/get-the-data.html> → "Download the
   Cumulative Data File" → **Stata** format. No registration required.
2. You get `GSS_stata.zip` (~46 MB). It contains `gss7224_r3.dta`.
3. Move the `.dta` to `experiment/datasets/GSS/`.

```bash
mkdir -p experiment/datasets/GSS
# after downloading GSS_stata.zip into that directory:
cd experiment/datasets/GSS && unzip GSS_stata.zip && ls -la gss7224_r3.dta
```

If NORC has since published Release 4+, the filename will differ (e.g.
`gss7226_r1.dta`). Point `GSS_DTA` in `config.py` at whatever you downloaded — but
note the paper's scope is `YEAR_MIN=2016`–`YEAR_MAX=2024`. A newer release adds
years beyond that window, so **results will shift unless you keep the year filter.**

> The GSS file needs `encoding='latin1'` — its value labels are not UTF-8 and
> `pyreadstat` raises `UnicodeDecodeError` without it. `config.py` sets this via
> `STATA_ENCODING`.

## WVS — World Values Survey Wave 7, v6.0, cross-national inverted CSV

1. Go to <https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp>.
2. Under **Statistical Data Files**, click
   `WVS Cross-National Wave 7 csv v6 0.zip`. Fill in the short form (name,
   institution, purpose) — access is free and the download starts automatically.
3. You get `F00011357-WVS_Cross-National_Wave_7_inverted_csv_v6_0.zip` (~21 MB).
4. Move the CSV to `experiment/datasets/WVS_raw/`.

```bash
mkdir -p experiment/datasets/WVS_raw
cd experiment/datasets/WVS_raw && unzip F00011357-*.zip && \
  ls -la WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv
```

Two things matter here:

- Take the **inverted** variant specifically. It is the one the WorldValuesBench
  probe set is keyed against, and the ordinal scales in `config.py` assume it.
- Take **v6.0**. v5.0 has different `D_INTERVIEW` values, which breaks the
  respondent-ID joins.

## WorldValuesBench (optional — not required)

The 16 WVS probe questions follow the
[WorldValuesBench](https://github.com/Demon702/WorldValuesBench) probe set and are
defined in `experiment/code/config.py`; no WorldValuesBench files are needed to run
anything here. If you want to compare against their pipeline, clone their
repository. Please cite the WorldValuesBench paper if you build on their question
set.

---

## Verifying the downloads

```bash
.venv/bin/python -c "
import pyreadstat, pandas as pd
_, m = pyreadstat.read_dta('experiment/datasets/GSS/gss7224_r3.dta',
                           metadataonly=True, encoding='latin1')
print('GSS:', f'{m.number_rows:,} rows x {m.number_columns:,} cols')  # expect 75,699 x 6,942
print('WVS:', len(pd.read_csv(
    'experiment/datasets/WVS_raw/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv',
    usecols=['B_COUNTRY'])), 'rows')                                  # expect 97,220
"
```

## Rebuilding the derived tables

```bash
cd experiment/code
../../.venv/bin/python build_dataset.py                                # GSS
LLM_FAULTS_DATASET=WVS ../../.venv/bin/python build_dataset_wvs.py     # WVS
../../.venv/bin/python sample_and_baseline.py                          # GSS
LLM_FAULTS_DATASET=WVS ../../.venv/bin/python sample_and_baseline.py   # WVS
```

Roughly 10 minutes. This regenerates `*_pilot_long.parquet`, `pilot_sample.parquet`,
`splits.parquet`, `human_dist.json`, and `baseline_preds.parquet`.

Everything except `wvs_pilot_long.parquet` is already included, so for GSS this only
matters if you are changing the sample construction. **For WVS, this step is how you
obtain `wvs_pilot_long.parquet`** — it is the file we do not redistribute. Once it
exists, `analyze_rq1_robustness.py` runs on WVS:

```bash
cd experiment/code && LLM_FAULTS_DATASET=WVS ../../.venv/bin/python analyze_rq1_robustness.py
```

`.gitignore` already excludes the rebuilt file, so you will not accidentally commit
it back.

The pipeline reads only 19 of the GSS file's 6,942 columns (`year`, `id`, 7
demographics, 10 questions) — over 99.7% of that 571 MB is never touched.

Then the weights appendix:

```bash
make weights
```

## Licensing

Redistribution of the raw files is restricted; that is why they are not here. See
[`../LICENSE-DATA`](../LICENSE-DATA) for the terms and required citations for both
survey programs.
