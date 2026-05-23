# Lower-Limb EMG Prediction From Exosuit Signals

This repository contains the code, notebooks, and experiment outputs for a machine learning study on predicting lower-limb EMG activity from exosuit sensor signals.

## Overview

The main goal of this project is to test whether exosuit measurements, primarily force and pressure signals, can be used to estimate muscle activation measured with EMG.

The study focuses on:
- 6 subjects
- 7 muscles: `Soleus`, `Tibialis`, `GastroMed`, `VastusMed`, `VastusLat`, `RectusFemoris`, `BicepsFemoris`
- 2 suit conditions:
- active assistance: `BAMon STANDARD 3x`
- passive / resistance off: `BAMoff`

## Research Questions

- Can exosuit sensor signals predict lower-limb EMG activity?
- Which model performs best in a subject-specific setting?
- Does the learned mapping generalize across subjects?
- Does the learned mapping transfer across active and passive conditions?

## Pipeline Summary

The main analysis pipeline:
- aligns EMG and SUIT signals in time
- resamples both signals to a common time grid
- splits the data into short overlapping windows
- predicts a window-level EMG target from suit features

The final main setup used:
- sampling frequency: `100 Hz`
- window size: `0.1 s`
- overlapping windows

Important note:
- the models predict a window-level EMG activity target
- they do not reconstruct the raw EMG waveform sample by sample

## Models Evaluated

- Ridge regression
- Random Forest
- XGBoost
- CNN-LSTM
- Deeper CNN-LSTM follow-up
- Multi-output CNN-LSTM across all 7 muscles

## Main Experiments

### 1. Subject-Specific Prediction

Models were trained and tested within the same subject.

Main result:
- all 7 muscles achieved positive mean `R2`
- XGBoost was the strongest overall model

Best subject-specific active results at `0.1 s`:
- `VastusLat`: `0.540`
- `GastroMed`: `0.492`
- `VastusMed`: `0.474`
- `Soleus`: `0.468`
- `Tibialis`: `0.389`
- `RectusFemoris`: `0.377`
- `BicepsFemoris`: `0.364`

### 2. Deeper CNN-LSTM Follow-Up

A deeper CNN-LSTM architecture was tested, especially for `Soleus` and `RectusFemoris`.

Main result:
- the deeper CNN-LSTM improved over the earlier smaller neural model
- it still did not outperform XGBoost

### 3. Cross-Subject Robustness

Cross-subject experiments used subject-level train/validation/test splits:
- train `[1, 2, 3, 4]`, test `[5]`, val `[6]`
- train `[1, 2, 5, 6]`, test `[3]`, val `[4]`
- train `[3, 4, 5, 6]`, test `[1]`, val `[2]`

This was evaluated for:
- active condition
- passive condition

Main result:
- performance dropped sharply compared with the subject-specific setting
- mean test `R2` became negative across muscles in both active and passive cross-subject runs

### 4. Multi-Output Neural Network

A shared CNN-LSTM was trained to predict all 7 muscles simultaneously in the cross-subject setting.

Main result:
- cross-subject performance remained poor
- the shared neural model did not solve the generalization problem

## Key Findings

- Exosuit sensor signals can estimate lower-limb EMG reasonably well within a subject.
- XGBoost was the best overall model in the final subject-specific setup.
- A deeper CNN-LSTM improved neural-model performance but did not surpass XGBoost.
- Cross-subject generalization was poor in both active and passive conditions.
- The mapping from exosuit signals to EMG appears strongly individual-specific.

## Practical Interpretation

The project suggests that exosuit force and pressure signals contain useful information about muscle activity, but the learned relationship is much more reliable within a person than across different people.

In other words:
- subject-specific modeling works
- cross-subject transfer remains difficult

## Repository Contents

This repository may include:
- analysis notebooks
- Python scripts for preprocessing, feature extraction, and modeling
- saved CSV summaries and fold results
- figures used for project reporting

## Suggested Paper Structure

- Introduction: motivation for predicting EMG from exosuit signals
- Methods: alignment, interpolation, windowing, feature extraction, models, evaluation
- Results: subject-specific performance, neural follow-up, cross-subject active/passive, multi-output model
- Discussion: interpretation of subject dependence and limited generalization
- Conclusion: feasibility within subject, weak transfer across subjects

## Current Status

The main subject-specific and cross-subject experiments are complete. The strongest project conclusion is that EMG prediction from exosuit signals is feasible within a subject, but generalization across subjects remains limited.

