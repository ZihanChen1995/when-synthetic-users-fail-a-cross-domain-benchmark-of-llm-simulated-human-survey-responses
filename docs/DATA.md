# Data reference

Every file shipped in this repository, what is in it, and how to load it.

If you only want to **benchmark your own model** against the same human ground
truth, you need three things and can ignore the rest:

| You want | Load |
|---|---|
| The personas + questions to prompt with | `experiment/datasets/<DS>/build/pilot_sample.parquet` |
| The human ground truth to score against | the `answer_code` column of that same file |
| Something to compare your score to | `experiment/datasets/<DS>/build/baseline_preds.parquet` and our four models' `llm_runs/` |

`examples/load_benchmark.py` does exactly this in ~40 lines. `<DS>` is `GSS` or `WVS`.

---

## 1. The two domains at a glance

| | GSS | WVS |
|---|---|---|
| Population | U.S. adults, 2016–2024 | 63 countries, Wave 7 (~2017–2022) |
| Respondents with full demographics | 14,704 | 91,760 |
| Attitude questions | 10 | 16 |
| Demographic variables in the prompt | 7 | 5 |
| Long-format rows | 85,898 (shipped) | 1,426,473 (**not shipped** — 1,985-row sample instead) |
| Evaluation sample (respondent × question) | 993 | 1,458 |
| Answer format | mostly 2–4 nominal options | ordinal scales (0–2, 1–4, 1–5, 1–10) |
| Model calls cached | 3,972 per model | 5,832 per model |

The two domains share **one protocol**. Everything downstream of `config.py` is
domain-agnostic; the domain is selected by the `LLM_FAULTS_DATASET` environment
variable.

---

## 2. Cached model outputs

The core asset: **50,076 model calls** (39,216 main + 10,860 robustness), with
the verbatim completion text retained.
Because `raw` is kept, the parser and every metric can be re-derived without
calling any model — which is what makes reproduction fully offline.

### `experiment/datasets/<DS>/build/llm_runs/<model>.jsonl`

One JSON object per line. One line = one (respondent, question, style, seed) call.

| Field | Type | Description |
|---|---|---|
| `model_name` | str | `claude-haiku-4.5`, `claude-sonnet-4.6`, `llama3.1-8b`, `llama3.3-70b` |
| `style` | str | `A` = pick one option; `C` = give a probability distribution |
| `seed` | int | `0` or `1` — two independent runs per cell, for the seed-noise floor |
| `respondent_id` | str | joins to `pilot_sample.parquet`. GSS: `<year>_<id>`; WVS: `<country>_<interview>` |
| `question_id` | str | GSS: `happy`, `trust`, … WVS: `Q48`, `Q106`, … |
| `human_answer` | int | **ground truth** — what the real respondent actually answered |
| `raw` | str | the model's verbatim completion text, unmodified |
| `pred` | int \| null | parsed answer code; `null` when parsing failed |
| `dist` | obj \| null | Style-C only: `{answer_code: probability}`. Always `null` for Style A |
| `valid` | bool | whether `raw` parsed into a legal answer code |

Row counts: 3,972 per model on GSS, 5,832 on WVS
(= respondent×question pairs × 2 styles × 2 seeds).

> **Gotcha — scoring Style C.** Style A puts the answer in `pred`. Style C leaves
> `pred` **null** and returns a distribution in `dist`; the individual prediction
> is the **argmax of `dist`**. Scoring Style C on `pred` alone silently gives you
> zero accuracy. See `predicted_code()` in `examples/load_benchmark.py`, which
> mirrors `metrics._pred_code`.

> **Two Style-C cells are unusable and are flagged with a dagger in the paper.**
> Llama-3.1-8B is 85% invalid on WVS (only n=440 parse) and 57% invalid on GSS.
> This is a reported finding (RQ4), not a pipeline defect. Filter on `valid`.

### `experiment/datasets/<DS>/build/robustness_runs/<model>.jsonl`

Prompt-perturbation runs for RQ4. **Same schema minus `dist`, plus two fields:**

| Field | Type | Description |
|---|---|---|
| `variant` | str | `order` = answer options reversed; `persona` = persona phrasing perturbed |
| `reverse` | bool | whether options were presented in reverse order |

Coverage: GSS has `order` + `persona` (1,986 each) for both Claude models.
WVS has an `order` spot-check only (1,458, one seed).

> **Gotcha — read `respondent_id` as a string.** WVS ids look like
> `"20_20070915"`, and Python's integer parser treats `_` as a digit separator, so
> a bare `pd.read_json(..., lines=True)` **silently** converts that id to the
> integer `2020070915`. It will then fail to join against the parquet tables — or,
> worse, join wrongly. Always pass `dtype={"respondent_id": str}`.
>
> The paper's own pipeline is unaffected: every script loads these files with
> `json.loads` line by line, never `pd.read_json`.

```python
import pandas as pd
runs = pd.read_json("experiment/datasets/GSS/build/llm_runs/claude-sonnet-4.6.jsonl",
                    lines=True, dtype={"respondent_id": str})   # <- dtype matters
# accuracy for style A, seed 0
a = runs[(runs.style == "A") & (runs.seed == 0) & runs.valid]
print((a.pred == a.human_answer).mean())
```

---

## 3. Derived human-response tables

### `<DS>/build/<ds>_pilot_long.parquet` — the full human dataset

Long format: one row per (respondent, question) with the respondent's demographics
repeated. GSS 85,898 × 17; WVS 1,426,473 × 14.

> **GSS ships in full. WVS does not.** The World Values Survey requires each user
> to accept its usage agreement before download, and its terms restrict
> redistribution of the microdata — so `wvs_pilot_long.parquet` is **not** in this
> repository. In its place you get
> **`wvs_pilot_long_SAMPLE.parquet`** — schema-identical, 1,985 rows, 126
> respondents (0.14% of the WVS sample), covering all 16 questions and all 63
> countries. It is for inspecting the format, not for analysis.
>
> To get the full table, accept the WVS agreement, download the data yourself, and
> rebuild it in ~10 minutes: see [RAW_DATA.md](RAW_DATA.md). This affects exactly
> one analysis script — `analyze_rq1_robustness.py` on WVS (Table 2). The other eight run
> fine without it, and that script's published output is included.

| Column | Type | Description |
|---|---|---|
| `dataset_id` | str | `GSS` or `WVS` |
| `respondent_id` | str | stable respondent key |
| `year` | int | GSS only — survey year (2016–2024) |
| `question_id` / `question_text` | str | item id and full wording as shown to the model |
| `topic` | str | `wellbeing`, `social_trust`, `moral_policy`, `institutions`, … |
| `ordinal` | bool | whether distance between options is meaningful (enables EMD) |
| *demographics* | int | GSS: `age`, `sex`, `race`, `degree`, `region`, `polviews`, `partyid`.<br>WVS: `agecat`, `sex`, `education`, `urbrural`, `country` |
| `answer_code` | int | the respondent's answer, as an integer code |
| `answer_text` | str | human-readable label for that code |
| `n_options` | int | number of legal options for this question |

Integer codes decode via `config.DEMOGRAPHIC_CODES` and `config.ANSWER_CODES`.

### `<DS>/build/splits.parquet` — respondent-level fit/eval split

`respondent_id`, `fold` ∈ {`fit`, `eval`}. Split is **by respondent**, so no
respondent appears in both. GSS: 7,255 fit / 7,449 eval. WVS: 45,748 / 46,012.
The non-LLM baselines are fit on `fit` and scored on `eval` — this is what makes
the baseline comparison honest.

### `<DS>/build/pilot_sample.parquet` — the evaluation sample

The `eval`-fold rows the models were actually asked about: **GSS 993, WVS 1,458**
(respondent × question). Same columns as `*_pilot_long` plus `fold`.
**This is the file to prompt from if you are benchmarking your own model.**

### `<DS>/build/baseline_preds.parquet` — non-LLM baseline predictions

| Column | Description |
|---|---|
| `respondent_id`, `question_id` | join keys |
| `baseline_pred` | predicted answer code |
| `baseline_dist` | predicted probability distribution |
| `baseline_backoff` | which cell the prediction came from: `full` (exact demographic cell), `coarse` (reduced cell), `marginal` (question marginal) |

The backoff hierarchy is why this baseline is strong: it uses the finest
demographic cell with enough support, and degrades gracefully.

### `<DS>/build/human_dist.json` — ground-truth target distributions

Per question: the observed human answer distribution, overall and per subgroup.

```json
{"abany": {"codes": [1, 2],
           "overall": {"1": 0.5456, "2": 0.4544},
           "n": 7368,
           "subgroups": {"sex=1": {"n": 3323, "dist": {"1": 0.5402, "2": 0.4598}}, ...}}}
```

Subgroup keys are `<var>=<code>`. Only cells with ≥ 100 humans are reported
(`config.MIN_CELL_HUMANS`). These are the targets for the distributional (JS
divergence) metrics.

### `<DS>/build/<ds>_pilot_meta.json` — build provenance

Scope, raw row counts, attrition at each filtering step, per-question option sets
and example counts. Read this to see exactly how the sample was constructed.

---

## 4. Analysis outputs

`experiment/analysis/<DS>/` — everything the paper reports.

| Paper element | File | Produced by |
|---|---|---|
| Main results table (16 rows) | `summary_table.csv`, `.md` | `summarize.py` |
| RQ1 robustness (learned baselines, distance, proper scoring) | `rq1_robustness.json` | `analyze_rq1_robustness.py` |
| RQ3 per-pair η² and CIs | `flattening_<model>.json` | `analyze_subgroups.py` (run via `summarize.py`) |
| RQ4 flip rates + seed-noise floor | `robustness.json` | `analyze_robustness.py` |
| RQ4 invalid-output bounds | `invalid_output_bounds.json` | `analyze_invalid_outputs.py` |
| Decision-impact analysis | `decision_impact.json` | `analyze_decision_impact.py` |
| Cramér's V, Δbase CIs, reweighted JS | `supplementary_metrics.json` | `analyze_supplementary.py` |
| Appendix: FDR + clustered bootstrap | `appendix_robustness.json` | `analyze_appendix_robustness.py` |
| Appendix: survey-weight check | `weight_robustness.json` | `analyze_weights.py` * |
| Per-model headline metrics | `pilot_metrics_<model>.json` | `metrics.py` (run via `summarize.py`) |

`*` `analyze_weights.py` is the only script needing the raw survey files — see
[RAW_DATA.md](RAW_DATA.md). Its published output is included.

### Main table columns → paper notation

| CSV column | Paper | Meaning |
|---|---|---|
| `indiv_acc` | Acc | individual-level accuracy |
| `baseline_acc` | Base | best non-LLM baseline on the same rows |
| `acc_minus_baseline` | Δbase | **the headline number.** Negative = the LLM loses to the baseline |
| `mean_js_div` | JS | Jensen–Shannon divergence from the human distribution |
| `median_stereotyping_index` | Δη² | over-determination: LLM η² minus human η². Positive = over-determines demographics |
| `n_pairs_ci_excl_0` / `n_pairs_total` | sig | question–group pairs whose CI excludes zero |
| `invalid_rate` | — | fraction of calls that failed to parse |

### Figures

Six per domain in `experiment/analysis/<DS>/`. Ten of them (five per domain) appear
in the paper; `fig_aggregate_js.png` is generated but unused.

---

## 5. Reference metadata

`experiment/datasets/WVS/country_codes.json` — the ISO-3166 numeric country code to
country-name mapping used by WVS `B_COUNTRY` (90 entries). `config.py` loads it at
import to render country labels in personas.

The 16 WVS probe questions and their scale endpoints follow the
[WorldValuesBench](https://github.com/Demon702/WorldValuesBench) probe set, for
comparability with prior work, and are defined in `config.py`.

---

## 6. What is *not* here, and why

We redistribute **our own** work — the model outputs, the metrics, the figures, the
code — plus the minimum derived survey material needed to verify the paper. We do
not redistribute anything the originating survey program restricts.

| Not shipped | Size | Why | To get it |
|---|---|---|---|
| `gss7224_r3.dta` | 571 MB | raw source file; public but large | [RAW_DATA.md](RAW_DATA.md) — no registration |
| `WVS_..._v6_0.csv` | 182 MB | raw source file | [RAW_DATA.md](RAW_DATA.md) — **requires the WVS form** |
| `wvs_pilot_long.parquet` | 9.2 MB | derived WVS microdata (91,760 respondents); **WVSA terms restrict redistribution** | rebuild in ~10 min after your own WVS download |

What this costs you: **one script**, `analyze_rq1_robustness.py` on WVS. The
other eight analysis scripts run without it, and `make verify` still reproduces all
16 rows of the main table exactly. `make reproduce` skips it on WVS automatically
and tells you so.

GSS is shipped in full because NORC distributes it publicly with no registration
and no usage agreement. The two programs are treated according to their own terms,
which is why the handling is asymmetric.

## 7. Licensing and citation

Data and figures are CC BY 4.0, except `WVS/build/`, which is non-profit
replication only (see [`../LICENSE-DATA`](../LICENSE-DATA)); code is MIT. The underlying GSS and WVS data
remain governed by their originators' terms — **if you use the derived tables you
must also cite GSS and WVS.** See [`../LICENSE-DATA`](../LICENSE-DATA) for the
required citations.
