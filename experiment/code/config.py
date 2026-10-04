"""Configuration for the cross-domain synthetic-users benchmark.

Two domains share ONE protocol. Select with env var LLM_FAULTS_DATASET:
  GSS  (default) : U.S. general social attitudes, General Social Survey 2016-2024.
  WVS            : cross-cultural values, World Values Survey Wave 7 (63 countries).

All downstream scripts (prompts, sampler, baseline, metrics, subgroups, summarize)
read from this module and are dataset-agnostic.
"""
import os
from pathlib import Path

DATASET = os.environ.get("LLM_FAULTS_DATASET", "GSS").upper()

# ---------------------------------------------------------------- paths (shared)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = EXPERIMENT_DIR / "datasets"
GSS_DTA = DATASETS_DIR / "GSS" / "gss7224_r3.dta"
WVS_CSV = DATASETS_DIR / "WVS_raw" / "WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv"
# ISO-3166 numeric country code -> country name, as used by WVS B_COUNTRY.
WVS_COUNTRY_CODES = DATASETS_DIR / "WVS" / "country_codes.json"
# analysis + build dirs are per-dataset so the two domains never overwrite each other
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis" / DATASET
BUILD_DIR = DATASETS_DIR / DATASET / "build"

STATA_ENCODING = "latin1"                                       # GSS value labels are latin1

# ---------------------------------------------------------------- scope (GSS only)
YEAR_MIN = 2016
YEAR_MAX = 2024

# ---------------------------------------------------------------- demographics used in the prompt
# order matters: this is the order shown to the model
_GSS_DEMOGRAPHIC_VARS = ["age", "sex", "race", "degree", "region", "polviews", "partyid"]

# human-readable phrasing of each demographic variable for persona prompts
_GSS_DEMOGRAPHIC_QUESTION_TEXT = {
    "age": "Age",
    "sex": "Sex",
    "race": "Race",
    "degree": "Highest degree earned",
    "region": "Region of the United States",
    "polviews": "Political views (liberal-conservative)",
    "partyid": "Political party affiliation",
}

# ---------------------------------------------------------------- subgroup axes for the subgroup analysis
# each must be categorical with a manageable number of levels
_GSS_SUBGROUP_VARS = ["sex", "race", "degree", "region", "polviews"]

# ---------------------------------------------------------------- attitude questions (the targets)
# key = GSS variable; text = question wording; category ordering follows the integer codes.
# `ordinal` marks scales where distance between options is meaningful (enables EMD/Wasserstein).
_GSS_QUESTIONS = {
    "happy":    {"text": "Taken all together, how would you say things are these days -- would you say that you are very happy, pretty happy, or not too happy?",
                 "topic": "wellbeing", "ordinal": True},
    "trust":    {"text": "Generally speaking, would you say that most people can be trusted or that you can't be too careful in dealing with people?",
                 "topic": "social_trust", "ordinal": False},
    "fair":     {"text": "Do you think most people would try to take advantage of you if they got a chance, or would they try to be fair?",
                 "topic": "social_trust", "ordinal": False},
    "helpful":  {"text": "Would you say that most of the time people try to be helpful, or that they are mostly just looking out for themselves?",
                 "topic": "social_trust", "ordinal": False},
    "cappun":   {"text": "Do you favor or oppose the death penalty for persons convicted of murder?",
                 "topic": "moral_policy", "ordinal": False},
    "grass":    {"text": "Do you think the use of marijuana should be made legal or not?",
                 "topic": "moral_policy", "ordinal": False},
    "abany":    {"text": "Please tell me whether or not you think it should be possible for a pregnant woman to obtain a legal abortion if the woman wants it for any reason.",
                 "topic": "moral_policy", "ordinal": False},
    "gunlaw":   {"text": "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?",
                 "topic": "moral_policy", "ordinal": False},
    "fefam":    {"text": "Do you agree or disagree with this statement: It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family?",
                 "topic": "gender_roles", "ordinal": True},
    "confinan": {"text": "How much confidence do you have in the people running banks and financial institutions -- a great deal, only some, or hardly any?",
                 "topic": "institutions", "ordinal": True},
}

# valid integer answer codes per question (from the GSS codebook value labels).
# Anything not in this set (string missing codes d/i/m/n/r/s/u/x/y/z) is treated as missing.
_GSS_ANSWER_CODES = {
    "happy":    {1: "very happy", 2: "pretty happy", 3: "not too happy"},
    "trust":    {1: "most people can be trusted", 2: "can't be too careful", 3: "depends"},
    "fair":     {1: "would take advantage of you", 2: "would try to be fair", 3: "depends"},
    "helpful":  {1: "try to be helpful", 2: "looking out for themselves", 3: "depends"},
    "cappun":   {1: "favor", 2: "oppose"},
    "grass":    {1: "should be legal", 2: "should not be legal"},
    "abany":    {1: "yes", 2: "no"},
    "gunlaw":   {1: "favor", 2: "oppose"},
    "fefam":    {1: "strongly agree", 2: "agree", 3: "disagree", 4: "strongly disagree"},
    "confinan": {1: "a great deal", 2: "only some", 3: "hardly any"},
}

# value labels for the demographic variables (from the codebook), used to render personas.
_GSS_DEMOGRAPHIC_CODES = {
    "sex":    {1: "male", 2: "female"},
    "race":   {1: "white", 2: "black", 3: "other"},
    "degree": {0: "less than high school", 1: "high school",
               2: "associate/junior college", 3: "bachelor's", 4: "graduate"},
    "region": {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    "polviews": {1: "extremely liberal", 2: "liberal", 3: "slightly liberal",
                 4: "moderate", 5: "slightly conservative", 6: "conservative",
                 7: "extremely conservative"},
    "partyid": {0: "strong Democrat", 1: "not very strong Democrat",
                2: "independent, close to Democrat", 3: "independent",
                4: "independent, close to Republican", 5: "not very strong Republican",
                6: "strong Republican", 7: "other party"},
    # age is continuous; handled specially (89 = "89 or older")
}

# any string value is a missing/non-response code in GSS; these are the ones observed.
MISSING_STR_CODES = set("dijmnprsuxyz")

RANDOM_SEED = 20260701      # fixed; no Date/random at runtime
MIN_CELL_HUMANS = 100       # only report subgroup cells with at least this many humans


# ======================================================================
# WVS (World Values Survey Wave 7) — cross-cultural values domain.
# Scope claim: WVS Wave 7 respondents across 63 countries (fielded ~2017-2022).
# Value questions are ordinal scales; demographics are country/education/sex/
# age-band/urban-rural. Negative raw codes = missing.
# Question texts and answer scales follow the WorldValuesBench probe set (for
# comparability with prior work).
# ======================================================================
_WVS_DEMOGRAPHIC_VARS = ["agecat", "sex", "education", "urbrural", "country"]

_WVS_DEMOGRAPHIC_QUESTION_TEXT = {
    "agecat": "Age group",
    "sex": "Sex",
    "education": "Education level",
    "urbrural": "Settlement type",
    "country": "Country",
}

# subgroup axes for the subgroup analysis. `country` is the primary cross-cultural
# axis; education/agecat/sex are secondary.
_WVS_SUBGROUP_VARS = ["country", "education", "agecat", "sex"]

# the 16 WVB probe value-questions present in the inverted v6.0 CSV, with WVB texts.
# scale endpoints (min..max) follow the WorldValuesBench probe set; all ordinal.
_WVS_QUESTIONS = {
    "Q48":  {"text": "On a scale of 1 to 10, 1 meaning 'None at all' and 10 meaning 'A great deal', how much freedom of choice and control over your life do you feel you have?",
             "topic": "wellbeing", "ordinal": True, "min": 1, "max": 10},
    "Q106": {"text": "On a scale of 1 to 10, 1 meaning 'Incomes should be made more equal' and 10 meaning 'There should be greater incentives for individual effort', where would you place your view?",
             "topic": "economic_values", "ordinal": True, "min": 1, "max": 10},
    "Q107": {"text": "On a scale of 1 to 10, 1 meaning 'Private ownership of business should be increased' and 10 meaning 'Government ownership of business should be increased', where would you place your view?",
             "topic": "economic_values", "ordinal": True, "min": 1, "max": 10},
    "Q108": {"text": "On a scale of 1 to 10, 1 meaning 'The government should take more responsibility to ensure that everyone is provided for' and 10 meaning 'People should take more responsibility to provide for themselves', where would you place your view?",
             "topic": "economic_values", "ordinal": True, "min": 1, "max": 10},
    "Q112": {"text": "On a scale of 1 to 10, 1 meaning 'No corruption at all' and 10 meaning 'Abundant corruption', how much corruption do you think there is in your country?",
             "topic": "institutions", "ordinal": True, "min": 1, "max": 10},
    "Q113": {"text": "On a scale of 1 to 4, 1 meaning 'None of them' and 4 meaning 'All of them', how many state authorities do you think are involved in corruption?",
             "topic": "institutions", "ordinal": True, "min": 1, "max": 4},
    "Q114": {"text": "On a scale of 1 to 4, 1 meaning 'None of them' and 4 meaning 'All of them', how many business executives do you think are involved in corruption?",
             "topic": "institutions", "ordinal": True, "min": 1, "max": 4},
    "Q121": {"text": "On a scale of 1 to 5, 1 meaning 'Very bad' and 5 meaning 'Very good', what impact do you think immigrants have on the development of your country?",
             "topic": "immigration", "ordinal": True, "min": 1, "max": 5},
    "Q122": {"text": "On a scale of 0 to 2, 0 meaning 'Disagree' and 2 meaning 'Agree', do you agree that immigration fills useful jobs in the labour market?",
             "topic": "immigration", "ordinal": True, "min": 0, "max": 2},
    "Q123": {"text": "On a scale of 0 to 2, 0 meaning 'Disagree' and 2 meaning 'Agree', do you agree that immigration increases the crime rate?",
             "topic": "immigration", "ordinal": True, "min": 0, "max": 2},
    "Q158": {"text": "On a scale of 1 to 10, 1 meaning 'Completely disagree' and 10 meaning 'Completely agree', how much do you agree that science and technology make our lives healthier, easier, and more comfortable?",
             "topic": "science", "ordinal": True, "min": 1, "max": 10},
    "Q159": {"text": "On a scale of 1 to 10, 1 meaning 'Completely disagree' and 10 meaning 'Completely agree', how much do you agree that because of science and technology there will be more opportunities for the next generation?",
             "topic": "science", "ordinal": True, "min": 1, "max": 10},
    "Q160": {"text": "On a scale of 1 to 10, 1 meaning 'Completely disagree' and 10 meaning 'Completely agree', how much do you agree that we depend too much on science and not enough on faith?",
             "topic": "science", "ordinal": True, "min": 1, "max": 10},
    "Q164": {"text": "On a scale of 1 to 10, 1 meaning 'Not at all important' and 10 meaning 'Very important', how important is God in your life?",
             "topic": "religion", "ordinal": True, "min": 1, "max": 10},
    "Q177": {"text": "On a scale of 1 to 10, 1 meaning 'Never justifiable' and 10 meaning 'Always justifiable', how justifiable do you think it is to claim government benefits to which you are not entitled?",
             "topic": "moral_justifiability", "ordinal": True, "min": 1, "max": 10},
    "Q178": {"text": "On a scale of 1 to 10, 1 meaning 'Never justifiable' and 10 meaning 'Always justifiable', how justifiable do you think it is to avoid paying a fare on public transport?",
             "topic": "moral_justifiability", "ordinal": True, "min": 1, "max": 10},
}

# WVS answer options = each integer on the ordinal scale, labelled by its position.
_WVS_ANSWER_CODES = {
    q: {i: str(i) for i in range(m["min"], m["max"] + 1)}
    for q, m in _WVS_QUESTIONS.items()
}

# demographic value labels for WVS personas. country is filled from
# country_codes.json at import time; the rest are fixed recodes applied in
# build_dataset_wvs.py.
_WVS_DEMOGRAPHIC_CODES = {
    "sex": {1: "male", 2: "female"},
    "agecat": {1: "16-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"},
    "education": {0: "Primary or less", 1: "Lower secondary",
                  2: "Upper/post-secondary", 3: "Tertiary"},
    "urbrural": {1: "urban", 2: "rural"},
    # country codes injected below
}


def _load_wvs_country_codes():
    import json
    return {int(k): v for k, v in json.loads(WVS_COUNTRY_CODES.read_text()).items()}


# ---------------------------------------------------------------- dispatch
if DATASET == "WVS":
    DEMOGRAPHIC_VARS = _WVS_DEMOGRAPHIC_VARS
    DEMOGRAPHIC_QUESTION_TEXT = _WVS_DEMOGRAPHIC_QUESTION_TEXT
    SUBGROUP_VARS = _WVS_SUBGROUP_VARS
    QUESTIONS = _WVS_QUESTIONS
    ANSWER_CODES = _WVS_ANSWER_CODES
    DEMOGRAPHIC_CODES = dict(_WVS_DEMOGRAPHIC_CODES)
    DEMOGRAPHIC_CODES["country"] = _load_wvs_country_codes()
    RESPONDENT_NOUN = "survey respondent"   # country is given in the demographics
elif DATASET == "GSS":
    DEMOGRAPHIC_VARS = _GSS_DEMOGRAPHIC_VARS
    DEMOGRAPHIC_QUESTION_TEXT = _GSS_DEMOGRAPHIC_QUESTION_TEXT
    SUBGROUP_VARS = _GSS_SUBGROUP_VARS
    QUESTIONS = _GSS_QUESTIONS
    ANSWER_CODES = _GSS_ANSWER_CODES
    DEMOGRAPHIC_CODES = _GSS_DEMOGRAPHIC_CODES
    RESPONDENT_NOUN = "U.S. survey respondent"
else:
    raise ValueError(f"unknown LLM_FAULTS_DATASET={DATASET!r} (use GSS or WVS)")
