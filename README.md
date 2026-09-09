# Subject-Specific Prediction of Lower-Limb EMG from R-HEXsuit Signals

This repository contains the analysis code, notebooks, summary outputs, and
manuscript source for predicting lower-limb surface electromyography (sEMG)
window targets from R-HEXsuit force and pressure signals.

## Study overview

The analysis includes:

- 6 participants;
- 7 lower-limb muscles: Soleus, Tibialis Anterior, Gastrocnemius Medialis,
  Vastus Medialis, Vastus Lateralis, Rectus Femoris, and Biceps Femoris;
- Earth and simulated lunar-gravity walking;
- active-resistance (`BAMon`) and actuation-off (`BAMoff`) conditions;
- training-mean, Ridge, Random Forest, and XGBoost regressors.

The models predict a window-level EMG-envelope target. They do not reconstruct
the raw EMG waveform sample by sample.

## Evaluation design

Every participant, muscle, gravity level, and suit condition is modelled
separately. Windows remain in chronological order. Before boundary purging,
approximately 64% of windows are allocated to training, 16% to validation, and
20% to testing. A 20-window purge is applied on both sides of each partition
boundary, removing 80 windows and creating an approximately 1.95 s signal-level
gap between adjacent partitions.

This is a **subject-specific, within-trial evaluation**. Training and testing use
different chronological sections of the same participant recording. The study
does not demonstrate prediction for an unseen participant, trial, session, or
day.

## Main findings

Across 168 participant--muscle--gravity--suit comparisons, XGBoost:

- outperformed the training-mean baseline in 147/168 comparisons (87.5%);
- outperformed Ridge in 131/168 comparisons (78.0%);
- outperformed Random Forest in 75/168 comparisons (44.6%).

The results support the use of nonlinear tree ensembles over constant and linear
baselines, but they do **not** demonstrate a consistent advantage for XGBoost
over Random Forest.

Median XGBoost test performance was descriptively higher under Earth than
simulated lunar gravity for every muscle in both suit states. No inferential
population-level gravity effect was tested. Performance varied substantially
between participants and included extreme negative test R-squared values in
low-variance target sections; these observations are retained and reported.

Participant 2 is used only for illustrative prediction traces. This participant
was selected objectively as the participant with the smallest median absolute
distance from the corresponding six-participant cell medians across all 28
muscle--gravity--suit combinations. Numerical conclusions use all six
participants.

## Repository structure

```text
notebooks/
  RHEXsuit_BAMon_Complete_Analysis.ipynb
  RHEXsuit_BAMoff_Complete_Analysis.ipynb

src/
  emg_exosuit_pipeline_colab.py
  generate_all_participant2_overleaf_figures.py

paper/
  EMG_RHEXsuit_revised_IEEE.tex
  figures/

results/
  BAMon/
  BAMoff/

```

Raw participant data are not included. Their use and redistribution remain
subject to the governance and ethics requirements of the originating study.

## Running the analysis 

1. Open the BAMon notebook in Google Colab and run all cells.
2. Confirm that `results_BAMon_complete_analysis` was created in Google Drive.
3. Run the BAMoff notebook using the same package versions.
4. Confirm that `results_BAMoff_complete_analysis` was created.
5. Run `generate_all_participant2_overleaf_figures.py` to export the manuscript
   figures without retraining the models.

The expected source-data folders in Colab are:

```text
/content/drive/MyDrive/data/emg_csv
/content/drive/MyDrive/data/suit2
```

If the folders differ, edit `EMG_FOLDER` and `SUIT_FOLDER` in the notebook
configuration cell.

## Results recommended for version control

The repository should contain derived, non-identifying outputs required to
verify the reported results, including:

- `model_comparison_subject_metrics.csv`;
- `model_comparison_summary.csv`;
- `paired_xgboost_comparisons.csv`;
- `paired_xgboost_summary.csv`;
- `xgboost_paper_summary.csv`;
- `chronological_split_audit.csv`;
- `xgboost_subject_metrics_with_diagnostics.csv`;
- `flagged_results_for_audit.csv`;
- `software_versions.json`;
- `analysis_configuration.json`;
- `feature_names_124.csv`.

Do not upload raw participant recordings, identifiers, credentials, tokens, or
restricted data.

## Manuscript

The IEEE manuscript source is available in `paper/`. The manuscript reports
results from all six participants, while participant-2 signal traces are
illustrative. Additional individual-muscle traces may be moved to supplementary
material to meet journal page limits.

## Important interpretation

This work is evidence of within-participant, within-trial feasibility after
paired EMG/exosuit calibration. It is not a calibration-free EMG estimator for a
new user. Cross-participant, cross-trial, and cross-session robustness require
separate evaluation.

