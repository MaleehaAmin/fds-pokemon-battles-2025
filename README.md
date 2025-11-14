#Feature Engineering + Ensemble Models (Kaggle Competition)

This repository contains the full reproducible code used for my submissions to the FDS Pokémon Battles Prediction 2025 Kaggle competition.

All modeling, data processing, and feature engineering logic lives here.
The public Kaggle notebook clones this repo and only calls the train_and_predict functions — no code is duplicated in the notebook, following competition fair-play rules.

#Structure
fds-pokemon-battles-2025/
├── requirements.txt
├── scripts/
│   ├── run_advstack.py
│   ├── run_ensemble_meta.py
│   └── run_ensemble_simple.py
└── src/
    └── fds_pokemon/
        ├── __init__.py
        ├── data.py
        ├── features.py
        ├── utils.py
        └── models/
            ├── __init__.py
            ├── ensemble_advstack.py
            ├── ensemble_calibrated_meta.py
            └── ensemble_calibrated_simple.py

This repo includes three submission pipelines:

Advanced Stacked Ensemble with Adversarial Weighting + TTA
→ ensemble_advstack.py

CV-Calibrated LightGBM Ensemble + Meta-Stacking
→ ensemble_calibrated_meta.py

Simple LightGBM Ensemble with Platt Calibration (Baseline)
→ ensemble_calibrated_simple.py

Each model generates its own CSV prediction file.

#Running Locally

Example:

python scripts/run_advstack.py
python scripts/run_ensemble_meta.py
python scripts/run_ensemble_simple.py


Or import programmatically:

from fds_pokemon.models.ensemble_advstack import train_and_predict
train_and_predict("train.jsonl", "test.jsonl", "submission.csv")

#Kaggle Notebook Usage

The Kaggle notebook:

Clones this repository.

Imports the package (fds_pokemon).

Calls each model’s train_and_predict function.

Produces three submission files:

submission_advstack.csv

submission_ensemble_platt_iso_stack.csv

submission_ensemble_platt.csv

This ensures full reproducibility and follows competition code-structure requirements.

