# Shark Attacks: A Global Exploratory Analysis

An exploratory analysis of shark attack incidents recorded worldwide, examining how encounters vary across geography, activity, and time. 

## Research Question

How have shark attack patterns shifted across geography, activity, and time, and what factors are most associated with fatal outcomes?

## Datasets

- `global_shark_attacks.csv` — sourced from the Global Shark Attack File (GSAF).

### Fields

- **Date:** calendar date of the incident (where known).
- **Year:** year of incident, used for temporal analysis.
- **Type:** classification of the encounter — e.g. *Unprovoked*, *Provoked*, *Boat*, *Sea Disaster*, *Invalid*.
- **Country / Area / Location:** geographic hierarchy from country down to specific beach or coordinate.
- **Activity:** what the victim was doing at the time (e.g. surfing, swimming, fishing, diving). Free-text field requiring normalization.
- **Sex / Age:** victim demographics, often partially missing.
- **Injury:** free-text description of injuries sustained.
- **Fatal (Y/N):** binary flag indicating whether the encounter was fatal.
- **Time:** time of day, frequently missing or approximate.
- **Species:** shark species involved, often reported secondhand and inconsistently named.
- **Source:** original reporting source (news article, medical record, investigator).

## Methodology

### 1. Cleaning & normalization (Python)
### 2. Exploratory analysis
### 3. Fatality association
### 4. Visualization (Python → HTML)

## Limitations

- **Reporting bias is the dominant confound.** Incidents in English-speaking, media-rich coastal regions are overrepresented; incidents in regions with less reporting infrastructure are systematically undercounted. Country-level comparisons should be read with this in mind.
- **Species identification is unreliable.** Many records identify species from eyewitness accounts under duress. Fine-grained species analysis is indicative rather than definitive.
- **Missingness is non-random.** Age, sex, and time-of-day are missing more often for older and non-Western records, which biases demographic conditional statistics. Missingness is quantified next to each result rather than imputed away.
- **Denominators are absent.** Raw incident counts cannot answer "how risky is surfing" without participation data, which GSAF does not provide. Activity-level risk claims are avoided accordingly.

## Attribution

Data courtesy of the [Global Shark Attack File](https://www.sharkattackfile.net/), Shark Research Institute. This project is not affiliated with GSAF or SRI, just a girl that likes to dive.
