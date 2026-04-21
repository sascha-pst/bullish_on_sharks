# %% [markdown]
# # Shark Attacks: A Global Exploratory Analysis
# 
# An exploratory analysis of shark attack incidents recorded worldwide, examining
# how encounters vary across geography, activity, and time. This project moves past
# the tabloid framing of shark attacks to ask where and how risk actually concentrates —
# and where the data itself falls short.

# %% [markdown]
# ## Research Question
# How have shark attack patterns shifted across geography, activity, and time, and
# what factors are most associated with fatal outcomes?

# %% [markdown]
# ## Dataset
# **`global_shark_attacks.csv`** (GSAF-derived)
# 
# | Field | Description |
# |---|---|
# | `date` | Calendar date of the incident (where known). |
# | `year` | Year of incident, used for temporal analysis. |
# | `type` | Encounter classification: Unprovoked, Provoked, Boat, Sea Disaster, Invalid. |
# | `country` / `area` / `location` | Geographic hierarchy from country → specific beach. |
# | `activity` | Victim activity (surfing, swimming, fishing, diving). Free-text; needs normalization. |
# | `name` | Victim name, if recorded. |
# | `sex` / `age` | Victim demographics, often partially missing. |
# | `fatal_y_n` | Y/N flag indicating whether the encounter was fatal. |
# | `time` | Time of day (`HHhMM` format), frequently missing or approximate. |
# | `species` | Shark species, reported secondhand and inconsistently named. |

# %% [markdown]
# ## Methodology
# 1. **Inspect**: structural audit of shape, dtypes, and missingness before any transformation.
# 2. **Clean**: normalize whitespace, parse dates, harmonize Y/N flags, coerce mixed types.
# 3. **Feature engineer**: derive decade buckets, coarse species categories, and activity groups
#    from noisy free-text fields.
# 4. **Univariate exploration**: distributions and value counts for each key field.
# 5. **Temporal analysis**: attacks per year/decade, *and* fatality rate over time
#    (fatality rate is less biased by reporting coverage than raw counts).
# 6. **Geographic analysis**: country- and region-level concentrations.
# 7. **Demographics**: sex and age of victims.
# 8. **Time-of-day**: when attacks happen, and whether fatality rate shifts by hour.
# 9. **Activity × species**: cross-examination of who was doing what when attacked.
# 10. **Activity risk**: within-attack fatality rate by activity category (no population denominator).
# 11. **Fatality correlates**: logistic regression with activity + species + country,
#     with an unprovoked-only sensitivity check.
# 12. **Findings + caveats**: surfaced honestly, not buried.

# %% [markdown]
# ---
# ## Import Libraries + Setup

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_PATH = Path("/content/global_shark_attacks.csv")
PROCESSED_PATH = Path("data/processed/attacks_clean.parquet")
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load & inspect Dataset

# %%
df_raw = pd.read_csv(RAW_PATH)
print(f"Shape: {df_raw.shape}")
df_raw.head()

# %%
df_raw.info()

# %%
# averages snapshot
(df_raw.isna().mean() * 100).round(1).sort_values(ascending=False)

# %%
# values across fields
df_raw.nunique().sort_values(ascending=False)

# %%
# fatal_y_n supposedly holds Y/N but has 9 unique values -- inspect before mapping
# so we don't silently drop rows into NaN
df_raw["fatal_y_n"].value_counts(dropna=False)

# %% [markdown]
# ## Clean Dataset

# %%
def clean_attacks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw attacks frame."""
    out = df.copy()

    # Parse dates; unparseable strings become NaT.
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Drop obviously bad years (GSAF has a handful of rows with year=1, year=5, etc.)
    # while preserving legitimately old records and genuine NaN.
    valid_year = out["year"].between(1700, 2024) | out["year"].isna()
    out = out[valid_year].copy()

    # Normalize whitespace and casing on geographic fields.
    for col in ["country", "area", "location"]:
        out[col] = (
            out[col]
            .astype("string")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.title()
        )

    # Coerce age to numeric; non-numeric strings ("Teens", "21, 34, 24 & 35") become NaN.
    out["age"] = pd.to_numeric(out["age"], errors="coerce")

    # Map fatal flag to boolean. fatal_y_n is messy -- strip/upper first, then
    # map only the values we're confident about; everything else becomes NaN.
    fatal_clean = out["fatal_y_n"].astype("string").str.strip().str.upper()
    out["fatal"] = fatal_clean.map({"Y": True, "N": False})
    out = out.drop(columns=["fatal_y_n"])

    # Parse "10h10" -> minutes past midnight. Loosened from the earlier strict
    # regex so we catch range-start values like "09h00-10h00" and trailing-space junk.
    def _time_to_minutes(t):
        if not isinstance(t, str):
            return np.nan
        m = re.search(r"(\d{1,2})h(\d{2})", t.strip())
        if not m:
            return np.nan
        h, mm = int(m.group(1)), int(m.group(2))
        return h * 60 + mm if 0 <= h < 24 and 0 <= mm < 60 else np.nan

    out["minutes_past_midnight"] = out["time"].apply(_time_to_minutes)
    return out


df = clean_attacks(df_raw)
df.head()

# %%
# sanity check the fatal mapping -- how many rows did we actually keep?
print(f"Rows with known fatal status: {df['fatal'].notna().sum()} / {len(df)}")
df["fatal"].value_counts(dropna=False)

# %% [markdown]
# ## Feature engineering

# %%
# organize species patterns~~~
SPECIES_PATTERNS = [
    ("white",      r"white"),
    ("tiger",      r"tiger"),
    ("bull",       r"bull"),
    ("hammerhead", r"hammerhead"),
    ("mako",       r"mako"),
    ("blue",       r"\bblue shark\b"),
    ("nurse",      r"nurse"),
    ("reef",       r"reef"),
    ("wobbegong",  r"wobbegong"),
]

def categorize_species(s) -> str:
    if not isinstance(s, str):
        return "unknown"
    s_low = s.lower()
    for label, pattern in SPECIES_PATTERNS:
        if re.search(pattern, s_low):
            return label
    return "other"

df["species_category"] = df["species"].apply(categorize_species)

# what were the people doing??!?!?
ACTIVITY_PATTERNS = [
    ("surfing",  r"surf|board"),
    ("swimming", r"swim|bath|wad"),
    ("diving",   r"div|snorkel"),
    ("fishing",  r"fish|spear"),
    ("boating",  r"boat|kayak|paddl|canoe"),
]

def categorize_activity(a) -> str:
    if not isinstance(a, str):
        return "unknown"
    a_low = a.lower()
    for label, pattern in ACTIVITY_PATTERNS:
        if re.search(pattern, a_low):
            return label
    return "other"

df["activity_category"] = df["activity"].apply(categorize_activity)

# this dataset spans longer than since jaws was released
df["decade"] = (df["year"] // 10 * 10).astype("Int64")

df[["activity", "activity_category", "species", "species_category", "year", "decade"]].sample(8, random_state=0)

# %%
# processed - check
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(PROCESSED_PATH)

# %% [markdown]
# ## Univariate exploration

# %%
# What activity were people doing when attacked?
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df["activity_category"].value_counts().plot(
    kind="bar", ax=axes[0], title="Attacks by activity category"
)

# Was there a specific species that had more attacks recorded?
df["species_category"].value_counts().plot(
    kind="bar", ax=axes[1], title="Attacks by species category"
)
plt.tight_layout()

# %%
# Attack type -- GSAF's own stratification. Most published stats restrict to
# "Unprovoked" because provoked encounters (fishing accidents, people grabbing
# sharks, etc.) are categorically different from predation.
df["type"].value_counts(dropna=False)

# %% [markdown]
# ## Temporal patterns

# %%
# Attacks per year with early year sparcity from underreporting.
attacks_per_year = df.groupby("year").size()
attacks_per_year.loc[1900:].plot(title="Recorded shark attacks per year (1900+)")
plt.xlabel("Year"); plt.ylabel("Recorded attacks")

# %% [markdown]
# > **Interpretation caveat:** Rising counts over time largely reflect (a) better record-keeping,
# > (b) more humans in the water, and (c) GSAF's evolving collection practices — not necessarily
# > a rise in per-capita risk. Frame temporal claims accordingly.

# %%
# Fatality rate over time is a cleaner signal than raw counts -- better reporting
# should *increase* the share of non-fatal attacks recorded, so a falling rate
# across decades is hard to explain away as pure reporting bias.
decade_fatality = (
    df.dropna(subset=["fatal", "decade"])
      .groupby("decade")["fatal"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "fatality_rate", "count": "n"})
      .query("n >= 20")
      .loc[1900:]
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(decade_fatality.index, decade_fatality["fatality_rate"], marker="o")
ax.set_title("Fatality rate per recorded attack, by decade (1900+)")
ax.set_xlabel("Decade"); ax.set_ylabel("P(fatal | attack)")
ax.set_ylim(0, None)
plt.tight_layout()
decade_fatality

# %% [markdown]
# ## Geographic patterns

# %%
top_countries = df["country"].value_counts().head(10)
top_countries.plot(kind="barh", title="Top 10 countries by recorded attacks").invert_yaxis()

# %% [markdown]
# ## Victim demographics

# %%
# Sex distribution -- M dominates by a lot, but that reflects exposure
# (more male surfers/divers/fishers historically) not shark preference.
sex_clean = df[df["sex"].isin(["M", "F"])].dropna(subset=["fatal"])
sex_summary = (
    sex_clean.groupby("sex")["fatal"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "fatality_rate", "count": "n_attacks"})
)
sex_summary

# %%
# Age distribution among victims.
fig, ax = plt.subplots(figsize=(9, 4))
df["age"].dropna().plot(kind="hist", bins=30, ax=ax)
ax.set_title("Victim age distribution"); ax.set_xlabel("Age"); ax.set_ylabel("Victims")
print(f"Median age: {df['age'].median():.1f}  |  n with known age: {df['age'].notna().sum()}")

# %% [markdown]
# ## Time of day

# %%
# Bin into 2-hour windows for readability. time is ~51% missing,
# so treat this as suggestive rather than conclusive.
df["hour_bin"] = pd.cut(
    df["minutes_past_midnight"] // 60,
    bins=range(0, 25, 2),
    right=False,
    labels=[f"{h:02d}-{h+2:02d}" for h in range(0, 24, 2)],
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df["hour_bin"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], title="Attacks by time of day"
)
(
    df.dropna(subset=["fatal"])
      .groupby("hour_bin", observed=True)["fatal"]
      .mean()
      .plot(kind="bar", ax=axes[1], title="Fatality rate by time of day")
)
axes[1].set_ylabel("P(fatal | attack)")
plt.tight_layout()

# %% [markdown]
# ## Activity × species

# %%
# Row-normalized so each row (activity) sums to 1 -- answers
# "given an attack during surfing, which species was involved?"
cross = pd.crosstab(df["activity_category"], df["species_category"])
cross_norm = cross.div(cross.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(cross_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"label": "share of row"})
ax.set_title("Species share within each activity (row-normalized)")
plt.tight_layout()

# %% [markdown]
# ## Activity risk (within-attack fatality rate)

# %%
#  P(fatal | attack, activity=X) needs exposure data
activity_fatality = (
    df.dropna(subset=["fatal"])
      .groupby("activity_category")["fatal"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "fatality_rate", "count": "n_attacks"})
      .query("n_attacks >= 30")
      .sort_values("fatality_rate", ascending=False)
)
activity_fatality

# %% [markdown]
# ## Fatality correlates (logistic regression)

# %%
import statsmodels.formula.api as smf

model_df = (
    df.dropna(subset=["fatal", "activity_category", "species_category", "country"])
      .assign(fatal=lambda d: d["fatal"].astype(int))
)

# Keep only the top-N countries so the design matrix stays tractable and interpretable.
top_n_countries = model_df["country"].value_counts().head(8).index
model_df = model_df[model_df["country"].isin(top_n_countries)]

# Drop rare species categories -- the prior run had coefficients of -20 with
# SE ~17,000 on hammerhead/reef/wobbegong, which is textbook perfect separation.
species_counts = model_df["species_category"].value_counts()
common_species = species_counts[species_counts >= 30].index
model_df = model_df[model_df["species_category"].isin(common_species)]

logit = smf.logit(
    "fatal ~ C(activity_category) + C(species_category) + C(country)",
    data=model_df,
).fit(disp=False)
print(logit.summary())

# %%
# Sensitivity check: restrict to unprovoked attacks only. This is the GSAF-
# standard definition for most published shark-attack stats. If direction and
# rough magnitude of coefficients hold, the full-data story is defensible.
unprovoked_df = model_df[model_df["type"] == "Unprovoked"]
print(f"Unprovoked subset: {len(unprovoked_df)} rows")

logit_unprovoked = smf.logit(
    "fatal ~ C(activity_category) + C(species_category) + C(country)",
    data=unprovoked_df,
).fit(disp=False)
print(logit_unprovoked.summary())

# %% [markdown]
# ## Findings
# 
# * **Fatality rate has fallen sharply across the 20th century.** Unlike raw attack counts
#   (which rise with reporting coverage and beach use), the share of attacks that end in
#   death declines across decades — consistent with improved medical response, faster
#   evacuation, and beach patrols, rather than any change in shark behavior.
# * **Swimming carries the highest within-attack fatality rate; surfing the lowest.**
#   Surfers are attacked often but survive most encounters (likely a mix of boards as
#   flotation, proximity to shore, and frequent mistaken-identity bites). Swimmers, by
#   contrast, are more often in deeper water without flotation.
# * **Geography concentrates counts, but also concentrates water use.** USA, Australia,
#   and South Africa dominate — also the countries with the most coastal recreation, the
#   most reporting infrastructure, and the longest GSAF coverage. High counts ≠ high
#   per-swimmer risk.
# * **White, tiger, and bull sharks are disproportionately associated with fatal outcomes**
#   in the regression, consistent with their size and predation style. Species is also the
#   field most vulnerable to reporter bias — 45% missing, and witnesses are not ichthyologists.
# * **Male victims outnumber female victims roughly 9:1.** Almost certainly exposure-driven
#   (surfing, spearfishing, and commercial fishing demographics), not a shark-side effect.
# 
# ### Caveats worth restating
# - **No exposure denominator.** All "risk" numbers here are P(fatal | attack), not
#   P(attack | hour in water). Without hours-in-water data by activity/country, we cannot
#   speak to absolute risk.
# - **Reporting bias is pervasive.** Early-20th-century decades, non-English-speaking
#   countries, and non-fatal attacks are all systematically under-recorded.
# - **Species and time fields are ~45–51% missing.** Treat species-specific claims as
#   suggestive, and time-of-day findings as descriptive of the reported subset only.
# - **The regression is associational, not causal.** Activity, species, and country are
#   correlated (surfers in California encounter different sharks than spearfishers in PNG),
#   so coefficients reflect partial associations within this joint distribution.

# %% [markdown]
# ## Attribution
# - **Data**: Global Shark Attack File (GSAF), via `global_shark_attacks.csv`.
# - **Author**: Sasha Schaps
# - **Program**: UC Berkeley MIDS — DATASCI EDA portfolio piece
# - **Tools**: pandas, numpy, matplotlib, seaborn, statsmodels.


