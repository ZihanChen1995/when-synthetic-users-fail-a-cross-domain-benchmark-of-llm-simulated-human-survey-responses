---
pretty_name: "When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey Responses"
license: cc-by-4.0
language:
- en
task_categories:
- multiple-choice
tags:
- synthetic-users
- survey-simulation
- silicon-sampling
- llm-evaluation
- computational-social-science
- general-social-survey
- stereotyping
size_categories:
- 10K<n<100K
configs:
- config_name: benchmark
  default: true
  data_files:
  - split: test
    path: data/benchmark/test.parquet
- config_name: questions
  data_files:
  - split: train
    path: data/questions/train.parquet
- config_name: model_outputs
  data_files:
  - split: train
    path: data/model_outputs/train.parquet
- config_name: robustness_outputs
  data_files:
  - split: train
    path: data/robustness_outputs/train.parquet
- config_name: human_responses
  data_files:
  - split: fit
    path: data/human_responses/fit.parquet
  - split: eval
    path: data/human_responses/eval.parquet
- config_name: human_distributions
  data_files:
  - split: train
    path: data/human_distributions/train.parquet
---

# When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey Responses

[![arXiv](https://img.shields.io/badge/arXiv-2607.26348-b31b1b.svg)](https://arxiv.org/abs/2607.26348)
[![GitHub](https://img.shields.io/badge/GitHub-code-black.svg)](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses)

Can a large language model stand in for a real survey respondent? This dataset is the
**General Social Survey (GSS)** half of the benchmark from
[*When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey
Responses*](https://arxiv.org/abs/2607.26348) (Chen, Zhu & Zheng, 2026). Each benchmark
row gives a model a real U.S. respondent's demographic profile and asks how that person
answered a GSS attitude question. The real answer is the ground truth, and a non-LLM
demographic baseline sets the bar to beat.

It includes everything needed to evaluate a new model with no other code: the exact
prompts, the human answers, the baseline's predictions, and all 23,832 cached outputs of
the four models in the paper.

**For more details, see the [GitHub repository](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses).** It has the analysis code that
reproduces every table and figure in the paper, the full documentation, and the
**World Values Survey (WVS) half** of the benchmark ([where to find the WVS
data](#where-is-the-world-values-survey-half)).

## Headline result

No model beats the simple demographic baseline at predicting what an individual
answered. All four models also **over-determine demographics**: they treat identity as
more predictive of attitudes than it is among real people (Δη² > 0).

| Model (single-answer prompt, 2 seeds) | Accuracy | Baseline | Δbase | Δη² (median) |
|---|---|---|---|---|
| Claude Haiku 4.5 | 0.5866 | 0.5891 | −0.0025 | +0.048 |
| Claude Sonnet 4.6 | 0.5629 | 0.5891 | −0.0262 | +0.081 |
| Llama 3.3-70B | 0.5685 | 0.5891 | −0.0206 | +0.051 |
| Llama 3.1-8B | 0.4960 | 0.5891 | −0.0932 | +0.026 |

Δbase = accuracy − baseline accuracy on the same rows. Δη² = the share of answer
variance explained by a demographic variable in the model's answers, minus the same
share among real respondents (median over question × demographic pairs). The paper reports the World Values Survey half, where
the gaps are larger.

## Quick start: evaluate your model

```python
from datasets import load_dataset

bench = load_dataset("ZihanChen/when-synthetic-users-fail", "benchmark", split="test")

correct = []
for row in bench:
    reply = my_model(row["prompt_a"]).strip()           # e.g. "A"
    key_to_code = {o["key"]: o["code"] for o in row["options"]}
    correct.append(key_to_code.get(reply) == row["human_answer"])

acc = sum(correct) / len(correct)
base = sum(r["baseline_pred"] == r["human_answer"] for r in bench) / len(bench)
print(f"accuracy {acc:.4f}   baseline {base:.4f}   Δbase {acc - base:+.4f}")
```

To compare with the paper's models on the same rows:

```python
outs = load_dataset("ZihanChen/when-synthetic-users-fail", "model_outputs", split="train").to_pandas()
print(outs[outs["valid"]].groupby(["model", "style"])["correct"].mean())
```

## Tables

Choose a table with the second argument of `load_dataset`.

| Table | Split(s) | Rows | What it is |
|---|---|---|---|
| `benchmark` (default) | `test` | 993 | One row per respondent × question, with prompts, human answer and baseline prediction. **Evaluate on this.** |
| `questions` | `train` | 10 | Question wording, topic and answer options |
| `model_outputs` | `train` | 15,888 | 4 models × 2 prompt styles × 2 seeds × 993 rows, with verbatim completion text |
| `robustness_outputs` | `train` | 7,944 | Reversed option order and reworded-persona prompts (Claude models) |
| `human_responses` | `fit`, `eval` | 85,898 | Every in-scope GSS answer: 14,704 respondents, split by respondent |
| `human_distributions` | `train` | 220 | Observed answer distributions per question, overall and per demographic subgroup |

### `benchmark`

| Column | Description |
|---|---|
| `respondent_id` | GSS respondent key, `<year>_<id>` (string) |
| `year` | survey year, 2016–2024 |
| `question_id`, `topic`, `ordinal`, `question_text` | the attitude question. `ordinal` is true when distance between options is meaningful |
| `age` | age in years (89 means 89 or older) |
| `sex`, `race`, `degree`, `region`, `polviews`, `partyid` | demographic labels as shown to the model; integer codes in the matching `*_code` columns |
| `options` | list of `{key, code, label}`. `key` is what the model is asked to reply with (`A`, `B`, …) and `code` is the answer it maps to |
| `n_options` | number of answer options |
| `human_answer`, `human_answer_label` | **ground truth**: what the respondent actually answered |
| `baseline_pred` | the non-LLM baseline's predicted answer code |
| `baseline_backoff` | which cell the baseline used: `full` (exact demographic cell), `coarse` (reduced cell) or `marginal` (question marginal) |
| `baseline_dist` | the baseline's predicted distribution, list of `{code, prob}` |
| `prompt_a` | **single-answer prompt** (Style A): reply with one option key |
| `prompt_b` | natural-language persona prompt (Style B), used for the robustness check |
| `prompt_c` | **probability prompt** (Style C): reply with JSON giving a probability for each option key |

### `model_outputs` and `robustness_outputs`

| Column | Description |
|---|---|
| `model` | `claude-haiku-4.5`, `claude-sonnet-4.6`, `llama3.1-8b`, `llama3.3-70b` |
| `style` | `A` (single answer), `B` (persona), `C` (probabilities) |
| `seed` | `0` or `1`: two independent runs per row |
| `respondent_id`, `question_id` | join keys to `benchmark` |
| `raw` | the model's verbatim reply |
| `valid` | whether `raw` parsed into a legal answer |
| `pred` | parsed answer code (Style A and B only; null for Style C) |
| `dist` | Style C only: parsed distribution, list of `{code, prob}` |
| `pred_final` | the model's single answer for any style (for Style C, the most probable option). **Score on this** |
| `human_answer` | ground truth |
| `correct` | `pred_final == human_answer` (null when invalid) |
| `variant`, `reverse` | robustness only: `order` (options shown in reverse, `reverse` = true) or `persona` (Style B wording) |

### `human_responses`

Same respondent, demographic and answer columns as `benchmark`, for all 14,704
respondents with complete demographics (GSS 2016–2024). The `fit` split (7,255
respondents) is what the paper's baseline was fit on. The `eval` split (7,449
respondents) contains the benchmark rows.

### `human_distributions`

`question_id`, `group_var` (`overall`, `sex`, `race`, `degree`, `region` or `polviews`),
`group_code`, `group_label`, `n` (number of respondents) and `dist` (list of
`{code, prob}`). Estimated on the full table. Only subgroups with at least 100
respondents are included. These are the targets for distribution-level metrics.

## Pitfalls

- **Train only on `human_responses` `fit`.** The benchmark respondents are in `eval`, so
  fitting or fine-tuning on `eval` leaks the answers.
- **Score Style C on `pred_final`, not `pred`.** `pred` is null for every Style C row, so
  scoring on it gives zero accuracy.
- **Filter on `valid`.** Llama 3.1-8B fails to return valid JSON for 57% of Style C rows.
  The paper reports this as a finding and flags that cell.
- **Compare like with like.** Δbase uses the baseline on exactly the rows that were
  scored. If you drop invalid rows, drop them from the baseline too.

## How it was built

- **Source:** GSS cumulative file 1972–2024 (Release 3), restricted to 2016–2024
  respondents with all 7 demographic variables present (14,704 respondents, 85,898
  respondent × question answers).
- **Questions:** 10 attitude items on wellbeing, social trust, moral and policy
  issues, gender roles, and confidence in institutions.
- **Split:** respondents are split 50/50 into `fit` and `eval` by a deterministic hash
  of their id, so no respondent is in both.
- **Benchmark sample:** about 100 rows per question, drawn by stratified sampling from
  `eval` (993 rows).
- **Baseline:** predicts the most common answer in the respondent's demographic cell in
  `fit`, backing off to coarser cells and finally the question marginal.
- **Models:** sampled at temperature 1.0, two seeds per prompt style. See the paper
  for the full protocol.

This release is generated by `scripts/build_hf_dataset.py` from data commit
`{{DATA_COMMIT}}` of the [GitHub repository](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses),
which also has the analysis code that reproduces every table and figure in the paper.

## Intended use and limitations

The benchmark is for **evaluating how well LLMs simulate survey respondents**, at the
level of individuals and of demographic groups. It is not meant for drawing conclusions
about real individuals or groups, and model outputs here should not be read as
anyone's views. Results cover U.S. adults, 10 questions and the tested prompts; other
populations, questions or prompting strategies may behave differently.

## Where is the World Values Survey half?

The paper's second domain uses the **World Values Survey (WVS) Wave 7**: 63 countries,
16 value questions, 1,458 benchmark rows and 23,328 cached model outputs. It is not in
this dataset because the WVS conditions of use (non-profit use only, no redistribution
of the data) are not compatible with an open CC BY 4.0 release.

- **Benchmark rows, baseline predictions and cached model outputs:** in the
  [GitHub repository](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses) under
  [`experiment/datasets/WVS/build/`](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses/tree/main/experiment/datasets/WVS/build).
  They are provided for non-profit replication only, under the WVSA conditions (see
  [`LICENSE-DATA`](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses/blob/main/LICENSE-DATA)).
- **Evaluating your own model on WVS:** clone the repository and run
  `python examples/evaluate_your_model.py export --domain WVS`, then `score`.
- **Full WVS microdata:** download it from the
  [WVS Association](https://www.worldvaluessurvey.org/WVSContents.jsp) (free
  registration). [`docs/RAW_DATA.md`](https://github.com/ZihanChen1995/when-synthetic-users-fail-a-cross-domain-benchmark-of-llm-simulated-human-survey-responses/blob/main/docs/RAW_DATA.md) explains how to
  rebuild the paper's tables from it.

## License and attribution

Released under **CC BY 4.0**. That covers our contribution: the derived tables, the
prompts, the cached model outputs and the metrics.

- **GSS:** the tables are derived from the General Social Survey, a project of NORC at
  the University of Chicago (principal investigators Michael Davern, Rene Bautista,
  Jeremy Freese, Pamela Herd and Stephen L. Morgan). Please also cite the GSS
  ([gss.norc.org](https://gss.norc.org/)).
- **Built with Llama.** The Llama outputs were generated with Llama 3.1-8B-Instruct and
  Llama 3.3-70B-Instruct, under the Llama 3.1 and Llama 3.3 Community Licenses
  (Copyright © Meta Platforms, Inc.). If you use these outputs to train a model that
  you distribute, those licenses require its name to begin with "Llama".
- **Claude outputs** were generated with Anthropic models. They are released for
  research and evaluation, not for training competing models.

## Citation

```bibtex
@article{chen2026syntheticusersfail,
  title   = {When Synthetic Users Fail: A Cross-Domain Benchmark of LLM-Simulated Human Survey Responses},
  author  = {Chen, Zihan and Zhu, Di and Zheng, Lei Nico},
  journal = {arXiv preprint arXiv:2607.26348},
  year    = {2026}
}
```
