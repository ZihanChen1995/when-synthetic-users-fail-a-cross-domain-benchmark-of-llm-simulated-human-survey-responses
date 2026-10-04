# Reproducing the paper

Every number, table, and figure in the paper is computed from the cached model
outputs in this repository. **No API key, no raw survey data, and no model calls
are needed.**

## Requirements

- **Python ≥ 3.11.** The pinned dependencies need it (pandas 3.x).
  macOS ships 3.9 as `python3`; using it fails at install time with a confusing
  resolver error. Tested on 3.12 and 3.14.
- ~1 GB free disk, no GPU, no network after setup.

## The three commands

```bash
make setup      # create .venv, install pinned deps          (~2 min)
make smoke      # integrity checks on this checkout           (~30 s)
make verify     # regenerate the main table, diff vs paper    (<1 min)
```

If your Python 3.12 is under a different name, pass it through:
`make setup PYTHON=/path/to/python3.12`.

`make verify` re-runs `summarize.py` for both domains, which recomputes the
per-model metrics and subgroup analysis from the cached model outputs, and
compares all 16 rows × 9 columns against the frozen reference in `expected/`.
On success:

```
VERIFIED -- 16 rows regenerated from cached model outputs and
matching the published table exactly.
```

## Full reproduction

```bash
make reproduce  # all 9 analysis scripts x 2 domains  (30-60 min)
make verify
```

Outputs are overwritten in place under `experiment/analysis/{GSS,WVS}/`.
The bootstrap scripts (`analyze_supplementary`, `analyze_appendix_robustness`,
`analyze_decision_impact`) take several minutes each.

To run one script by hand:

```bash
cd experiment/code
LLM_FAULTS_DATASET=WVS ../../.venv/bin/python analyze_decision_impact.py
```

`LLM_FAULTS_DATASET` selects the domain (`GSS` default, or `WVS`). Every script
downstream of `config.py` is domain-agnostic.

### Scripts that need data not included here

- `analyze_weights.py` (survey-weight appendix) re-reads the raw survey files,
  because the weights (`wtssps`, `W_WEIGHT`) are not in the build tables. It is
  excluded from `make reproduce`; its output is included. To run it, get the raw
  data ([RAW_DATA.md](RAW_DATA.md)) and run `make weights`.
- `analyze_rq1_robustness.py` (Table 2 — RQ1 robustness) fits learned baselines
  over the **full** human sample. On GSS that table is included and the script
  runs. On WVS it is not, because the World Values Survey restricts
  redistribution of its microdata; `make reproduce` skips it on WVS and says so.
  Its WVS output is included at `experiment/analysis/WVS/rq1_robustness.json`. To
  run it, download WVS and rebuild the table (~10 min, see [RAW_DATA.md](RAW_DATA.md)).

Of the 18 runs in `make reproduce` (9 scripts × 2 domains), 17 run from this
repository alone; the remaining one is the WVS case above.

## Mapping outputs to the paper

`experiment/analysis/<DS>/summary_table.csv` holds the 16 rows of the main
table. Column mapping:

| CSV column | Paper |
|---|---|
| `indiv_acc` | Acc |
| `baseline_acc` | Base |
| `acc_minus_baseline` | Δbase |
| `mean_js_div` | JS |
| `median_stereotyping_index` | Δη² |
| `n_pairs_ci_excl_0` / `n_pairs_total` | sig |

The full file-by-file map is in [DATA.md §4](DATA.md#4-analysis-outputs).

---

# The full pipeline

| Stage | What | Needs |
|---|---|---|
| 1 | Build the analysis tables (`build_dataset*.py`, `sample_and_baseline.py`) | raw survey files — see [RAW_DATA.md](RAW_DATA.md) |
| 2 | Model inference (`run_inference.py`, `run_robustness.py`) | model access |
| 3–4 | Analysis and figures (`make reproduce`) | nothing beyond this repository |

The outputs of Stages 1 and 2 are included, so Stages 3–4 run on their own.

## Stage 2 — model inference (optional)

To run new inference, implement `APIClient.generate()` in
`experiment/code/inference.py` for the API you use, or pass `--backend vllm` to
use a local OpenAI-compatible server. The runner scripts
(`run_gss_pilot.sh`, `run_llama_pilot.sh`, `run_wvs_pilot.sh`,
`run_robustness_pilot.sh`) read model identifiers from `HAIKU_MODEL_ID`,
`SONNET_MODEL_ID`, `LLAMA8B_MODEL_ID`, and `LLAMA70B_MODEL_ID`.

Models: Claude Haiku 4.5, Claude Sonnet 4.6, Llama-3.1-8B-Instruct,
Llama-3.3-70B-Instruct. Decoding: temperature 1.0, `max_tokens` 100, two
independent runs per cell (seed 0 and seed 1). Fresh runs will not match the
cached outputs bit-for-bit, since decoding is sampled.

---

# Known caveats

1. **Two Style-C cells are unusable** and are flagged with a dagger in the paper:
   Llama-8B is 85% invalid on WVS (n=440) and 57% invalid on GSS. This is a
   reported finding (RQ4), not a pipeline defect.
2. **Re-running inference will not reproduce the cache exactly.** Decoding is
   sampled at two seeds by design; the paper's stability analysis quantifies this
   (a seed-noise flip-rate floor of 1.1–2.6% on GSS).
3. **Pinned versions are the tested working set**, not a strict floor. Older
   pandas/numpy will very likely work; the parquet files require `pyarrow`.
