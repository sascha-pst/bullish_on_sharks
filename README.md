# Shark Attacks: A Global Exploratory Analysis

An exploratory analysis of shark attack incidents recorded worldwide, examining how encounters vary across geography, activity, and time. This project moves past the tabloid framing of shark attacks to ask where and how risk actually concentrates — and where the data itself falls short.

## Research Question
How have shark attack patterns shifted across geography, activity, and time, and what factors are most associated with fatal outcomes?

## Datasets

- global_shark_attacks.csv (Source: Kaggle)

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


## Attribution

Data courtesy of the [Global Shark Attack File](https://www.sharkattackfile.net/), Shark Research Institute. This project is not affiliated with GSAF or SRI, just a girl that likes to dive.
