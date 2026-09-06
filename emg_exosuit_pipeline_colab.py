"""
GitHub-ready pipeline for predicting lower-limb EMG from exosuit signals.

This file keeps the stable core of the project:
- data parsing and loading
- alignment and window building
- handcrafted feature extraction
- subject-specific model evaluation
- cross-subject XGBoost evaluation

Notebook-only debugging cells, Colab mount/install commands, and unfinished
exploratory transfer checks are intentionally left out.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.stats import kurtosis, pearsonr, skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers
from xgboost import XGBRegressor


CONDITION_MAP = {
    "bamoff": "BAMoff",
    "bamon": "BAMon STANDARD 3x",
    "bamonstandard3x": "BAMon STANDARD 3x",
    "bamonstnd3x": "BAMon STANDARD 3x",
    "walk": "Walk",
    "walknoexo": "Walk",
    "noexo": "NOexo",
}

MUSCLE_ALIASES = {
    "Soleus": ["soleus", "sol"],
    "Tibialis": ["tibialis", "tibialisanterior", "ta"],
    "GastroMed": ["gastromed", "gastrocnemiusmed", "gastrocnemiusmedialis", "gm"],
    "VastusMed": ["vastusmed", "vastusmedialis", "vm"],
    "VastusLat": ["vastuslat", "vastuslateralis", "vl"],
    "RectusFemoris": ["rectusfemoris", "rf"],
    "BicepsFemoris": ["bicepsfemoris", "bf"],
}

ALL_MUSCLES = list(MUSCLE_ALIASES.keys())


@dataclass
class ExperimentConfig:
    emg_folder: str
    suit_folder: str
    results_root: str
    conditions: List[str] = field(default_factory=lambda: ["BAMon STANDARD 3x"])
    target_muscles: List[str] = field(default_factory=lambda: ALL_MUSCLES.copy())
    cnn_lstm_target_muscles: List[str] = field(
        default_factory=lambda: ["Soleus", "RectusFemoris"]
    )
    suit_features: List[str] = field(
        default_factory=lambda: ["forcefront", "forceback", "pressurefront", "pressureback"]
    )
    target_fs: float = 100.0
    window_seconds: float = 0.1
    window_overlap: float = 0.5
    min_trial_samples: int = 30
    target_aggregation: str = "rms"
    seed: int = 42
    run_handcrafted_models: bool = True
    run_cnn_lstm: bool = True
    ridge_alphas: List[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    rf_grid: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 1},
            {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 3},
        ]
    )
    xgb_grid: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {
                "n_estimators": 400,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            {
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            },
        ]
    )
    cnn_lstm_config: Dict[str, object] = field(
        default_factory=lambda: {
            "run_tag": "cnnlstm_deep",
            "conv_filters": [64, 128, 256],
            "conv_layers_per_block": 2,
            "kernel_size": 3,
            "pool_size": 2,
            "spatial_dropout": 0.15,
            "lstm_units": [128],
            "dense_units": [128, 64],
            "dropout": 0.30,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 60,
            "patience": 8,
            "bidirectional": False,
        }
    )
    cross_subject_splits: List[Dict[str, object]] = field(
        default_factory=lambda: [
            {"name": "split_1", "train": [1, 2, 3, 4], "test": [5], "val": [6]},
            {"name": "split_2", "train": [1, 2, 5, 6], "test": [3], "val": [4]},
            {"name": "split_3", "train": [3, 4, 5, 6], "test": [1], "val": [2]},
        ]
    )


@dataclass
class WindowDataset:
    X_features: np.ndarray
    X_windows: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    trial_ids: np.ndarray
    window_times: np.ndarray
    feature_names: List[str]
    window_samples: int


def canon(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def norm_condition(condition: str) -> str:
    return CONDITION_MAP.get(canon(condition), str(condition).strip())


def make_trial_id(
    subject: int,
    gravity: str,
    condition: str,
    repetition: int | None = None,
) -> str:
    """Create a condition key and, when available, retain the repetition."""
    base = f"S{int(subject)}_{gravity.lower()}_{condition}"
    return base if repetition is None else f"{base}_trial{int(repetition):02d}"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def format_decimal_tag(value: float) -> str:
    return str(value).replace(".", "p")


def build_run_name(config: ExperimentConfig) -> str:
    muscle_tag = (
        "all7"
        if len(config.target_muscles) == len(ALL_MUSCLES)
        else "_".join(m.lower() for m in config.target_muscles)
    )
    fs_tag = f"fs{int(config.target_fs)}"
    window_tag = f"w{format_decimal_tag(config.window_seconds)}"
    mode_tag: List[str] = []
    if config.run_handcrafted_models:
        mode_tag.append("tree")
    if config.run_cnn_lstm:
        mode_tag.append(str(config.cnn_lstm_config.get("run_tag", "cnnlstm")))
    return f"{muscle_tag}_{fs_tag}_{window_tag}_{config.target_aggregation}_{'_'.join(mode_tag)}"


def parse_emg_meta(path: str) -> Tuple[int, str, str] | None:
    base = os.path.basename(path)
    base = re.sub(r"\s*\(1\)", "", base).replace(" 2", "")
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    subject_match = re.search(r"(\d+)", parts[0])
    if not subject_match:
        return None
    subject = int(subject_match.group(1))

    gravity_idx, gravity = None, None
    for i, token in enumerate(parts):
        if token.lower() in {"earth", "moon"}:
            gravity_idx, gravity = i, token.lower()
            break
    if gravity_idx is None:
        return None

    all_idx = next((i for i, token in enumerate(parts) if token.lower() == "all"), len(parts))
    condition_raw = "_".join(parts[gravity_idx + 1 : all_idx])
    if not condition_raw:
        return None
    return subject, gravity, norm_condition(condition_raw)


def parse_suit_meta(path: str) -> Tuple[int, str, str] | None:
    base = os.path.basename(path)
    base = re.sub(r"\s*\(1\)", "", base)
    base = base.replace("SUIT_data_", "")
    stem = os.path.splitext(base)[0]
    key = canon(stem)

    subject_match = re.search(r"(?:^|_)(?:sbj|s)(\d+)(?:_|$)", stem, flags=re.IGNORECASE)
    if not subject_match:
        return None
    subject = int(subject_match.group(1))

    gravity_match = re.search(r"(earth|moon)", stem, flags=re.IGNORECASE)
    if not gravity_match:
        return None
    gravity = gravity_match.group(1).lower()

    if "bamoff" in key:
        condition = "BAMoff"
    elif "bamonstandard3x" in key or "bamonstnd3x" in key or "bamon" in key:
        condition = "BAMon STANDARD 3x"
    elif "walk" in key and "noexo" in key:
        condition = "Walk"
    elif "noexo" in key:
        condition = "NOexo"
    else:
        return None

    return subject, gravity, condition


def normalize_suit_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for column in df.columns:
        key = canon(column)
        if key in {"time", "tsuitseg", "tsuit", "tsuits", "tsuitsec"}:
            rename[column] = "time"
        elif key in {"forcefront", "frontforce", "ffront"}:
            rename[column] = "forcefront"
        elif key in {"forceback", "backforce", "fback", "bforce"}:
            rename[column] = "forceback"
        elif key in {"pressurefront", "frontpressure", "pfront", "presfront"}:
            rename[column] = "pressurefront"
        elif key in {"pressureback", "backpressure", "pback", "presback"}:
            rename[column] = "pressureback"
    return df.rename(columns=rename).copy()


def ensure_emg_time(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" in df.columns:
        return df.copy()
    out = df.copy()
    for column in out.columns:
        if canon(column) == "time":
            return out.rename(columns={column: "Time"})
    return out.rename(columns={out.columns[0]: "Time"})


def resolve_muscle_col(df: pd.DataFrame, muscle: str) -> str:
    columns_by_key = {canon(column): column for column in df.columns}
    for alias in MUSCLE_ALIASES.get(muscle, []) + [muscle]:
        key = canon(alias)
        if key in columns_by_key:
            return columns_by_key[key]
    raise KeyError(f"Could not find EMG column for muscle: {muscle}")


def load_emg_data(emg_folder: str, conditions: Sequence[str]) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(emg_folder, "*.csv")))
    allow = {norm_condition(item) for item in conditions}
    frames = []
    skipped = 0
    trial_counts: Dict[Tuple[int, str, str], int] = {}

    for path in files:
        meta = parse_emg_meta(path)
        if meta is None:
            skipped += 1
            continue

        subject, gravity, condition = meta
        if allow and condition not in allow:
            continue

        df = ensure_emg_time(pd.read_csv(path))
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"]).copy()
        if df.empty:
            skipped += 1
            continue

        df["Subject"] = int(subject)
        df["Gravity"] = gravity
        df["Condition"] = condition
        key = (subject, gravity, condition)
        trial_counts[key] = trial_counts.get(key, 0) + 1
        df["TrialID"] = make_trial_id(subject, gravity, condition, trial_counts[key])
        df["SourceFile"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No EMG trials were loaded. Check emg_folder and condition filtering: {conditions}"
        )

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded EMG rows={len(out):,}, trials={out['TrialID'].nunique()}, skipped_files={skipped}")
    return out


def load_suit_data(
    suit_folder: str, conditions: Sequence[str], suit_features: Sequence[str]
) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(suit_folder, "*.csv")))
    allow = {norm_condition(item) for item in conditions}
    required = ["time"] + list(suit_features)
    frames = []
    skipped = 0
    trial_counts: Dict[Tuple[int, str, str], int] = {}

    for path in files:
        meta = parse_suit_meta(path)
        if meta is None:
            skipped += 1
            continue

        subject, gravity, condition = meta
        if allow and condition not in allow:
            continue

        df = normalize_suit_columns(pd.read_csv(path))
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            skipped += 1
            continue

        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=required).copy()
        if df.empty:
            skipped += 1
            continue

        df["Subject"] = int(subject)
        df["Gravity"] = gravity
        df["Condition"] = condition
        key = (subject, gravity, condition)
        trial_counts[key] = trial_counts.get(key, 0) + 1
        df["TrialID"] = make_trial_id(subject, gravity, condition, trial_counts[key])
        df["SourceFile"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No SUIT trials were loaded. Check suit_folder and required columns: {required}"
        )

    out = pd.concat(frames, ignore_index=True)
    print(f"Loaded SUIT rows={len(out):,}, trials={out['TrialID'].nunique()}, skipped_files={skipped}")
    return out


def build_aligned_trial_series(
    suit_trial: pd.DataFrame,
    emg_trial: pd.DataFrame,
    muscle_col: str,
    suit_features: Sequence[str],
    fs: float,
    min_trial_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_suit = suit_trial["time"].to_numpy(float)
    t_emg = emg_trial["Time"].to_numpy(float)
    t0 = max(np.min(t_suit), np.min(t_emg))
    t1 = min(np.max(t_suit), np.max(t_emg))
    if t1 <= t0:
        raise ValueError("No common time interval between SUIT and EMG.")

    t_grid = np.arange(t0, t1, 1.0 / fs)
    if len(t_grid) < min_trial_samples:
        raise ValueError("Aligned trial is too short.")

    x_parts = [
        np.interp(t_grid, t_suit, suit_trial[feature].to_numpy(float)).astype(np.float32)
        for feature in suit_features
    ]
    x = np.stack(x_parts, axis=1)
    y = np.interp(t_grid, t_emg, emg_trial[muscle_col].to_numpy(float)).astype(np.float32)
    return t_grid, x, y


def safe_skew(signal: np.ndarray) -> float:
    if np.std(signal) < 1e-12:
        return 0.0
    return float(skew(signal, bias=False))


def safe_kurtosis(signal: np.ndarray) -> float:
    if np.std(signal) < 1e-12:
        return 0.0
    return float(kurtosis(signal, fisher=True, bias=False))


def zero_crossings(signal: np.ndarray) -> int:
    return int(np.sum(np.diff(np.signbit(signal)) != 0))


def signal_entropy(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=float))
    total = np.sum(values)
    if total <= 1e-12:
        return 0.0
    probs = values / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def compute_time_features(signal: np.ndarray) -> Dict[str, float]:
    q25, q75 = np.percentile(signal, [25, 75])
    peaks, _ = find_peaks(signal)
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "range": float(np.max(signal) - np.min(signal)),
        "median": float(np.median(signal)),
        "iqr": float(q75 - q25),
        "abs_mean": float(np.mean(np.abs(signal))),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "energy": float(np.mean(signal**2)),
        "skew": safe_skew(signal),
        "kurtosis": safe_kurtosis(signal),
        "peaks": float(len(peaks)),
        "zero_crossings": float(zero_crossings(signal)),
    }


def compute_diff_features(signal: np.ndarray) -> Dict[str, float]:
    diff = np.diff(signal) if len(signal) > 1 else np.array([0.0])
    return {
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "diff_min": float(np.min(diff)),
        "diff_max": float(np.max(diff)),
        "diff_abs_mean": float(np.mean(np.abs(diff))),
        "diff_rms": float(np.sqrt(np.mean(diff**2))),
        "diff_energy": float(np.mean(diff**2)),
        "diff_zero_crossings": float(zero_crossings(diff)),
    }


def compute_fft_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    if len(spectrum) <= 1:
        return {
            "fft_mean": 0.0,
            "fft_std": 0.0,
            "fft_max": 0.0,
            "fft_energy": 0.0,
            "fft_dom_freq": 0.0,
            "fft_centroid": 0.0,
            "fft_bandwidth": 0.0,
            "fft_entropy": 0.0,
        }

    spectrum = spectrum[1:]
    freqs = freqs[1:]
    if np.sum(spectrum) <= 1e-12:
        return {
            "fft_mean": 0.0,
            "fft_std": 0.0,
            "fft_max": 0.0,
            "fft_energy": 0.0,
            "fft_dom_freq": 0.0,
            "fft_centroid": 0.0,
            "fft_bandwidth": 0.0,
            "fft_entropy": 0.0,
        }

    dom_idx = int(np.argmax(spectrum))
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    bandwidth = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum))
    )
    return {
        "fft_mean": float(np.mean(spectrum)),
        "fft_std": float(np.std(spectrum)),
        "fft_max": float(np.max(spectrum)),
        "fft_energy": float(np.mean(spectrum**2)),
        "fft_dom_freq": float(freqs[dom_idx]),
        "fft_centroid": centroid,
        "fft_bandwidth": bandwidth,
        "fft_entropy": signal_entropy(spectrum),
    }


def extract_window_features(
    window: np.ndarray, feature_names: Sequence[str], fs: float
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    feature_name_set = set(feature_names)
    for channel_idx, channel_name in enumerate(feature_names):
        signal = window[:, channel_idx].astype(float)
        for name, value in compute_time_features(signal).items():
            row[f"{channel_name}__{name}"] = value
        for name, value in compute_diff_features(signal).items():
            row[f"{channel_name}__{name}"] = value
        for name, value in compute_fft_features(signal, fs).items():
            row[f"{channel_name}__{name}"] = value

    if {"forcefront", "forceback"}.issubset(feature_name_set):
        ff = window[:, feature_names.index("forcefront")].astype(float)
        fb = window[:, feature_names.index("forceback")].astype(float)
        row["force_balance_mean"] = float(np.mean(ff - fb))
        row["force_balance_std"] = float(np.std(ff - fb))

    if {"pressurefront", "pressureback"}.issubset(feature_name_set):
        pf = window[:, feature_names.index("pressurefront")].astype(float)
        pb = window[:, feature_names.index("pressureback")].astype(float)
        row["pressure_balance_mean"] = float(np.mean(pf - pb))
        row["pressure_balance_std"] = float(np.std(pf - pb))

    return row


def aggregate_target(y_window: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(np.mean(y_window))
    if mode == "rms":
        return float(np.sqrt(np.mean(y_window**2)))
    if mode == "max":
        return float(np.max(np.abs(y_window)))
    raise ValueError(f"Unsupported target_aggregation: {mode}")


def build_window_dataset(
    emg_df: pd.DataFrame,
    suit_df: pd.DataFrame,
    muscle: str,
    suit_features: Sequence[str],
    fs: float,
    window_seconds: float,
    overlap: float,
    target_aggregation: str,
    min_trial_samples: int,
) -> WindowDataset:
    muscle_col = resolve_muscle_col(emg_df, muscle)
    common_trials = sorted(set(emg_df["TrialID"].unique()) & set(suit_df["TrialID"].unique()))
    window_samples = max(4, int(round(window_seconds * fs)))
    stride = max(1, int(round(window_samples * (1.0 - overlap))))

    feature_rows, raw_windows, targets, subjects, trial_ids, times = [], [], [], [], [], []

    for trial_id in common_trials:
        emg_trial = emg_df[emg_df["TrialID"] == trial_id].sort_values("Time")
        suit_trial = suit_df[suit_df["TrialID"] == trial_id].sort_values("time")

        emg_trial = emg_trial.dropna(subset=["Time", muscle_col]).drop_duplicates(subset=["Time"])
        suit_trial = suit_trial.dropna(subset=["time"] + list(suit_features)).drop_duplicates(
            subset=["time"]
        )

        if len(emg_trial) < min_trial_samples or len(suit_trial) < min_trial_samples:
            continue

        try:
            t_grid, x_trial, y_trial = build_aligned_trial_series(
                suit_trial=suit_trial,
                emg_trial=emg_trial,
                muscle_col=muscle_col,
                suit_features=suit_features,
                fs=fs,
                min_trial_samples=min_trial_samples,
            )
        except ValueError:
            continue

        if len(x_trial) < window_samples:
            continue

        subject = int(emg_trial["Subject"].iloc[0])
        for start in range(0, len(x_trial) - window_samples + 1, stride):
            end = start + window_samples
            x_window = x_trial[start:end]
            y_window = y_trial[start:end]
            feature_rows.append(extract_window_features(x_window, suit_features, fs))
            raw_windows.append(x_window)
            targets.append(aggregate_target(y_window, target_aggregation))
            subjects.append(subject)
            trial_ids.append(trial_id)
            times.append(float(t_grid[end - 1]))

    if not feature_rows:
        raise RuntimeError(f"No aligned windows were built for muscle {muscle}.")

    features_df = pd.DataFrame(feature_rows)
    x_features = features_df.to_numpy(dtype=np.float32)
    x_windows = np.asarray(raw_windows, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    subjects_arr = np.asarray(subjects, dtype=int)
    trial_ids_arr = np.asarray(trial_ids)
    times_arr = np.asarray(times, dtype=float)

    print(
        f"{muscle}: windows={len(x_features)}, subjects={sorted(np.unique(subjects_arr).tolist())}, "
        f"window_samples={window_samples}, features={x_features.shape[1]}"
    )

    return WindowDataset(
        X_features=x_features,
        X_windows=x_windows,
        y=y,
        subjects=subjects_arr,
        trial_ids=trial_ids_arr,
        window_times=times_arr,
        feature_names=features_df.columns.tolist(),
        window_samples=window_samples,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    mae = float(mean_absolute_error(true_flat, pred_flat))
    rmse = float(np.sqrt(mean_squared_error(true_flat, pred_flat)))
    r2 = float(r2_score(true_flat, pred_flat))
    corr = (
        np.nan
        if np.std(true_flat) < 1e-12 or np.std(pred_flat) < 1e-12
        else float(pearsonr(true_flat, pred_flat)[0])
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def choose_subject_specific_masks(
    trial_ids_subject: np.ndarray,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    purge_windows: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split with unused gaps between overlapping signal sections.

    The purge prevents neighbouring, highly correlated windows from straddling
    the train/validation/test boundaries. With the default 0.1 s windows and
    50% overlap, 20 windows correspond to approximately one second.
    """
    trial_ids_subject = np.asarray(trial_ids_subject)
    train_mask = np.zeros(len(trial_ids_subject), dtype=bool)
    val_mask = np.zeros(len(trial_ids_subject), dtype=bool)
    test_mask = np.zeros(len(trial_ids_subject), dtype=bool)

    for trial_id in pd.unique(trial_ids_subject):
        idx = np.flatnonzero(trial_ids_subject == trial_id)
        n_trial = len(idx)
        if n_trial < 3:
            train_mask[idx] = True
            continue

        n_test = max(1, int(np.ceil(n_trial * test_fraction)))
        n_remaining = n_trial - n_test
        if n_remaining < 2:
            n_test = 1
            n_remaining = n_trial - n_test

        n_val = max(1, int(np.ceil(n_remaining * val_fraction)))
        if n_remaining - n_val < 1:
            n_val = max(1, n_remaining - 1)

        test_idx = idx[-n_test:]
        val_start = n_trial - n_test - n_val
        val_end = n_trial - n_test
        val_idx = idx[val_start:val_end]
        train_idx = idx[:val_start]

        gap = min(
            max(1, int(purge_windows)),
            max(1, len(train_idx) // 4),
            max(1, len(val_idx) // 4),
        )
        if len(train_idx) > gap:
            train_idx = train_idx[:-gap]
        if len(val_idx) > 2 * gap:
            val_idx = val_idx[gap:-gap]
        if len(test_idx) > gap:
            test_idx = test_idx[gap:]

        if len(train_idx) == 0:
            train_idx = idx[:1]
            remaining_idx = idx[1:]
            if len(remaining_idx) == 1:
                val_idx = remaining_idx
                test_idx = remaining_idx
            else:
                half = max(1, len(remaining_idx) // 2)
                val_idx = remaining_idx[:half]
                test_idx = remaining_idx[half:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    if not np.any(test_mask):
        test_mask[-1] = True
        train_mask[-1] = False
    if not np.any(val_mask):
        candidates = np.flatnonzero(train_mask)
        if len(candidates) > 1:
            val_mask[candidates[-1]] = True
            train_mask[candidates[-1]] = False
    if not np.any(train_mask):
        candidates = np.flatnonzero(~test_mask)
        if len(candidates) > 0:
            train_mask[candidates[0]] = True
            val_mask[candidates[0]] = False
    return train_mask, val_mask, test_mask


def scale_feature_matrix(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    return (
        x_scaler.transform(x_train).astype(np.float32),
        y_scaler.transform(y_train).astype(np.float32),
        x_scaler.transform(x_val).astype(np.float32),
        y_scaler.transform(y_val).astype(np.float32),
        x_scaler.transform(x_test).astype(np.float32),
        x_scaler,
        y_scaler,
    )


def scale_sequence_windows(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    n_features = x_train.shape[-1]
    x_scaler = StandardScaler().fit(x_train.reshape(-1, n_features))
    y_scaler = StandardScaler().fit(y_train)

    def transform(x: np.ndarray) -> np.ndarray:
        x_s = x_scaler.transform(x.reshape(-1, n_features)).reshape(x.shape)
        return x_s.astype(np.float32)

    return (
        transform(x_train),
        y_scaler.transform(y_train).astype(np.float32),
        transform(x_val),
        y_scaler.transform(y_val).astype(np.float32),
        transform(x_test),
        x_scaler,
        y_scaler,
    )


def save_prediction_artifact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    times: np.ndarray,
    out_dir: str,
    title: str,
    trial_ids: np.ndarray | None = None,
) -> None:
    ensure_dir(out_dir)
    times = np.asarray(times, dtype=float).reshape(-1)
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if trial_ids is None:
        pred_df = pd.DataFrame(
            {"time_absolute_s": times, "true_target": y_true, "pred_target": y_pred}
        )
        pred_df = pred_df.sort_values("time_absolute_s").reset_index(drop=True)
        pred_df["time_s"] = pred_df["time_absolute_s"] - pred_df["time_absolute_s"].min()
        pred_df["plot_time_s"] = pred_df["time_s"]
    else:
        pred_df = pd.DataFrame(
            {
                "trial_id": np.asarray(trial_ids).astype(str).reshape(-1),
                "time_absolute_s": times,
                "true_target": y_true,
                "pred_target": y_pred,
            }
        )
        pred_df = pred_df.sort_values(["trial_id", "time_absolute_s"]).reset_index(drop=True)
        pred_df["time_s"] = pred_df["time_absolute_s"] - pred_df.groupby("trial_id")[
            "time_absolute_s"
        ].transform("min")

        # Keep trials visually separated so points from different trials are never joined.
        pred_df["plot_time_s"] = np.nan
        offset = 0.0
        for trial_id, trial_df in pred_df.groupby("trial_id", sort=False):
            idx = trial_df.index
            pred_df.loc[idx, "plot_time_s"] = trial_df["time_s"].to_numpy() + offset
            if len(trial_df) > 1:
                step = float(np.median(np.diff(np.sort(trial_df["time_s"].to_numpy()))))
                if not np.isfinite(step) or step <= 0:
                    step = 0.5
            else:
                step = 0.5
            offset = float(pred_df.loc[idx, "plot_time_s"].max() + max(0.5, 5.0 * step))

    pred_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    if "trial_id" in pred_df.columns:
        for trial_id, trial_df in pred_df.groupby("trial_id", sort=False):
            axes[0].plot(
                trial_df["plot_time_s"],
                trial_df["true_target"],
                marker=".",
                linewidth=1.0,
                label=f"true {trial_id}",
            )
            axes[0].plot(
                trial_df["plot_time_s"],
                trial_df["pred_target"],
                marker=".",
                linewidth=1.0,
                linestyle="--",
                label=f"pred {trial_id}",
            )
    else:
        axes[0].plot(pred_df["plot_time_s"], pred_df["true_target"], label="true", linewidth=1.5)
        axes[0].plot(
            pred_df["plot_time_s"], pred_df["pred_target"], label="predicted", linewidth=1.5
        )
    axes[0].set_title("Held-out windows over time; trials kept separate")
    axes[0].set_xlabel("Time (s), separated by trial")
    axes[0].set_ylabel("Window target")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(pred_df["true_target"], pred_df["pred_target"], s=8, alpha=0.4)
    lo = float(min(np.min(pred_df["true_target"]), np.min(pred_df["pred_target"])))
    hi = float(max(np.max(pred_df["true_target"]), np.max(pred_df["pred_target"])))
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="black")
    axes[1].set_title("Predicted vs true")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Predicted")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prediction_overlay.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_importance_artifact(
    model_name: str, model: object, feature_names: Sequence[str], out_dir: str
) -> None:
    ensure_dir(out_dir)
    if model_name == "ridge":
        importances = np.abs(model.coef_).reshape(-1)
    elif model_name in {"random_forest", "xgboost"} and hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    else:
        return

    df = pd.DataFrame({"feature": list(feature_names), "importance": importances}).sort_values(
        "importance", ascending=False
    )
    df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    top_df = df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title("Top 20 features")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_training_history(history: tf.keras.callbacks.History, out_dir: str) -> None:
    ensure_dir(out_dir)
    pd.DataFrame(history.history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    if "loss" not in history.history:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CNN-LSTM training curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fit_best_ridge(
    x_train_s: np.ndarray,
    y_train_s: np.ndarray,
    x_val_s: np.ndarray,
    y_val_s: np.ndarray,
    ridge_alphas: Sequence[float],
    seed: int,
) -> Tuple[Ridge, Dict[str, float]]:
    best_model, best_alpha, best_score = None, None, np.inf
    for alpha in ridge_alphas:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(x_train_s, y_train_s.reshape(-1))
        val_pred = model.predict(x_val_s).reshape(-1, 1)
        score = mean_absolute_error(y_val_s.reshape(-1), val_pred.reshape(-1))
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_model = model
    return best_model, {"alpha": float(best_alpha), "val_mae_scaled": float(best_score)}


def fit_best_random_forest(
    x_train_s: np.ndarray,
    y_train: np.ndarray,
    x_val_s: np.ndarray,
    y_val: np.ndarray,
    rf_grid: Sequence[Dict[str, float]],
    seed: int,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    best_model, best_params, best_score = None, None, np.inf
    for params in rf_grid:
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
        model.fit(x_train_s, y_train.reshape(-1))
        val_pred = model.predict(x_val_s).reshape(-1, 1)
        score = mean_absolute_error(y_val.reshape(-1), val_pred.reshape(-1))
        if score < best_score:
            best_score = score
            best_params = params
            best_model = model
    meta = dict(best_params)
    meta["val_mae"] = float(best_score)
    return best_model, meta


def fit_best_xgboost(
    x_train_s: np.ndarray,
    y_train: np.ndarray,
    x_val_s: np.ndarray,
    y_val: np.ndarray,
    xgb_grid: Sequence[Dict[str, float]],
    seed: int,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    best_model, best_params, best_score = None, None, np.inf
    for params in xgb_grid:
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            reg_lambda=1.0,
            n_jobs=2,
            **params,
        )
        model.fit(x_train_s, y_train.reshape(-1), eval_set=[(x_val_s, y_val.reshape(-1))], verbose=False)
        val_pred = model.predict(x_val_s).reshape(-1, 1)
        score = mean_absolute_error(y_val.reshape(-1), val_pred.reshape(-1))
        if score < best_score:
            best_score = score
            best_params = params
            best_model = model
    meta = dict(best_params)
    meta["val_mae"] = float(best_score)
    return best_model, meta


def refit_ridge(alpha: float, x_fit: np.ndarray, y_fit: np.ndarray, seed: int) -> Ridge:
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(x_fit, y_fit.reshape(-1))
    return model


def refit_random_forest(
    params: Dict[str, float], x_fit: np.ndarray, y_fit: np.ndarray, seed: int
) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    model.fit(x_fit, y_fit.reshape(-1))
    return model


def refit_xgboost(
    params: Dict[str, float], x_fit: np.ndarray, y_fit: np.ndarray, seed: int
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        reg_lambda=1.0,
        n_jobs=2,
        **params,
    )
    model.fit(x_fit, y_fit.reshape(-1), verbose=False)
    return model


def build_cnn_lstm_model(input_shape: Tuple[int, int], config: Dict[str, object]) -> models.Model:
    conv_filters = config["conv_filters"]
    if isinstance(conv_filters, int):
        conv_filters = [conv_filters]
    lstm_units = config["lstm_units"]
    if isinstance(lstm_units, int):
        lstm_units = [lstm_units]
    dense_units = config["dense_units"]
    if isinstance(dense_units, int):
        dense_units = [dense_units]

    inputs = layers.Input(shape=input_shape, name="window_input")
    x = inputs
    for block_idx, filters in enumerate(conv_filters, start=1):
        for conv_idx in range(int(config.get("conv_layers_per_block", 1))):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=int(config["kernel_size"]),
                padding="same",
                use_bias=False,
                name=f"conv_block{block_idx}_{conv_idx + 1}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_block{block_idx}_{conv_idx + 1}")(x)
            x = layers.Activation("relu", name=f"relu_block{block_idx}_{conv_idx + 1}")(x)
        x = layers.MaxPooling1D(
            pool_size=int(config["pool_size"]),
            padding="same",
            name=f"pool_block{block_idx}",
        )(x)
        spatial_dropout = float(config.get("spatial_dropout", 0.0))
        if spatial_dropout > 0:
            x = layers.SpatialDropout1D(spatial_dropout, name=f"spatial_dropout_block{block_idx}")(x)

    for lstm_idx, units in enumerate(lstm_units, start=1):
        return_sequences = lstm_idx < len(lstm_units)
        lstm_layer = layers.LSTM(int(units), return_sequences=return_sequences, name=f"lstm_block_{lstm_idx}")
        if bool(config.get("bidirectional", False)):
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_block_{lstm_idx}")(x)
        else:
            x = lstm_layer(x)
        if float(config["dropout"]) > 0:
            x = layers.Dropout(float(config["dropout"]), name=f"dropout_lstm_block_{lstm_idx}")(x)

    for dense_idx, units in enumerate(dense_units, start=1):
        x = layers.Dense(int(units), activation="relu", name=f"dense_block_{dense_idx}")(x)
        if float(config["dropout"]) > 0:
            x = layers.Dropout(float(config["dropout"]), name=f"dropout_dense_block_{dense_idx}")(x)

    outputs = layers.Dense(1, name="emg_output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=str(config.get("run_tag", "cnn_lstm_window_regressor")))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(config["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_cnn_lstm(
    x_train_s: np.ndarray,
    y_train_s: np.ndarray,
    x_val_s: np.ndarray,
    y_val_s: np.ndarray,
    cnn_config: Dict[str, object],
    out_dir: str,
    seed: int,
) -> Tuple[models.Model, Dict[str, float]]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_cnn_lstm_model((x_train_s.shape[1], x_train_s.shape[2]), cnn_config)

    model_dir = ensure_dir(out_dir)
    model_ckpt_path = os.path.join(model_dir, "best_model.keras")
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cnn_config, handle, indent=2)

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cnn_config["patience"]),
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            model_ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        x_train_s,
        y_train_s,
        validation_data=(x_val_s, y_val_s),
        epochs=int(cnn_config["epochs"]),
        batch_size=int(cnn_config["batch_size"]),
        verbose=0,
        callbacks=cb,
    )
    save_training_history(history, model_dir)

    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
        if "val_loss" in history.history
        else len(history.history["loss"])
    )
    meta = {
        "epochs_ran": int(len(history.history["loss"])),
        "best_epoch": best_epoch,
        "best_val_loss": float(
            np.min(history.history["val_loss"])
            if "val_loss" in history.history
            else np.min(history.history["loss"])
        ),
        "parameter_count": int(model.count_params()),
    }

    with open(os.path.join(model_dir, "model_summary.txt"), "w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    return model, meta


def run_subject_specific_handcrafted_for_muscle(
    data: WindowDataset, muscle: str, results_dir: str, config: ExperimentConfig
) -> pd.DataFrame:
    rows = []
    mode_dir = ensure_dir(os.path.join(results_dir, "subject_specific", muscle))

    for subject in np.unique(data.subjects):
        subject_mask = data.subjects == subject
        x_subject = data.X_features[subject_mask]
        y_subject = data.y[subject_mask]
        trial_ids_subject = data.trial_ids[subject_mask]
        times_subject = data.window_times[subject_mask]
        if len(x_subject) < 10:
            continue

        train_mask, val_mask, test_mask = choose_subject_specific_masks(trial_ids_subject)
        if train_mask.sum() < 2 or val_mask.sum() < 1 or test_mask.sum() < 1:
            continue

        x_train, y_train = x_subject[train_mask], y_subject[train_mask]
        x_val, y_val = x_subject[val_mask], y_subject[val_mask]
        x_test, y_test = x_subject[test_mask], y_subject[test_mask]
        times_test = times_subject[test_mask]
        trial_ids_test = trial_ids_subject[test_mask]

        x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, _, y_scaler = scale_feature_matrix(
            x_train, y_train, x_val, y_val, x_test
        )
        x_fit_s = np.vstack([x_train_s, x_val_s])
        y_fit_s = np.vstack([y_train_s, y_val_s])
        x_fit_scaled = np.vstack([x_train_s, x_val_s])
        y_fit_raw = np.vstack([y_train, y_val])

        ridge_model, ridge_meta = fit_best_ridge(
            x_train_s, y_train_s, x_val_s, y_val_s, config.ridge_alphas, config.seed
        )
        ridge_final = refit_ridge(ridge_meta["alpha"], x_fit_s, y_fit_s, config.seed)
        ridge_pred = y_scaler.inverse_transform(ridge_final.predict(x_test_s).reshape(-1, 1))
        ridge_metrics = compute_metrics(y_test, ridge_pred)
        ridge_dir = ensure_dir(os.path.join(mode_dir, f"subject_{int(subject)}", "ridge"))
        save_prediction_artifact(
            y_test,
            ridge_pred,
            times_test,
            ridge_dir,
            f"{muscle} | Ridge | subject {int(subject)}",
            trial_ids_test,
        )
        save_importance_artifact("ridge", ridge_final, data.feature_names, ridge_dir)
        rows.append(
            {
                "evaluation_mode": "subject_specific",
                "model": "ridge",
                "muscle": muscle,
                "test_subject": int(subject),
                "val_subject": int(subject),
                "n_train": int(len(x_train)),
                "n_val": int(len(x_val)),
                "n_test": int(len(x_test)),
                **ridge_metrics,
                **{f"meta_{k}": v for k, v in ridge_meta.items()},
            }
        )

        rf_model, rf_meta = fit_best_random_forest(
            x_train_s, y_train, x_val_s, y_val, config.rf_grid, config.seed
        )
        rf_params = {k: rf_meta[k] for k in ["n_estimators", "max_depth", "min_samples_leaf"]}
        rf_final = refit_random_forest(rf_params, x_fit_scaled, y_fit_raw, config.seed)
        rf_pred = rf_final.predict(x_test_s).reshape(-1, 1)
        rf_metrics = compute_metrics(y_test, rf_pred)
        rf_dir = ensure_dir(os.path.join(mode_dir, f"subject_{int(subject)}", "random_forest"))
        save_prediction_artifact(
            y_test,
            rf_pred,
            times_test,
            rf_dir,
            f"{muscle} | Random Forest | subject {int(subject)}",
            trial_ids_test,
        )
        save_importance_artifact("random_forest", rf_final, data.feature_names, rf_dir)
        rows.append(
            {
                "evaluation_mode": "subject_specific",
                "model": "random_forest",
                "muscle": muscle,
                "test_subject": int(subject),
                "val_subject": int(subject),
                "n_train": int(len(x_train)),
                "n_val": int(len(x_val)),
                "n_test": int(len(x_test)),
                **rf_metrics,
                **{f"meta_{k}": v for k, v in rf_meta.items()},
            }
        )

        xgb_model, xgb_meta = fit_best_xgboost(
            x_train_s, y_train, x_val_s, y_val, config.xgb_grid, config.seed
        )
        xgb_params = {
            k: xgb_meta[k]
            for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
        }
        xgb_final = refit_xgboost(xgb_params, x_fit_scaled, y_fit_raw, config.seed)
        xgb_pred = xgb_final.predict(x_test_s).reshape(-1, 1)
        xgb_metrics = compute_metrics(y_test, xgb_pred)
        xgb_dir = ensure_dir(os.path.join(mode_dir, f"subject_{int(subject)}", "xgboost"))
        save_prediction_artifact(
            y_test,
            xgb_pred,
            times_test,
            xgb_dir,
            f"{muscle} | XGBoost | subject {int(subject)}",
            trial_ids_test,
        )
        save_importance_artifact("xgboost", xgb_final, data.feature_names, xgb_dir)
        rows.append(
            {
                "evaluation_mode": "subject_specific",
                "model": "xgboost",
                "muscle": muscle,
                "test_subject": int(subject),
                "val_subject": int(subject),
                "n_train": int(len(x_train)),
                "n_val": int(len(x_val)),
                "n_test": int(len(x_test)),
                **xgb_metrics,
                **{f"meta_{k}": v for k, v in xgb_meta.items()},
            }
        )
    return pd.DataFrame(rows)


def run_subject_specific_cnn_lstm_for_muscle(
    data: WindowDataset, muscle: str, results_dir: str, config: ExperimentConfig
) -> pd.DataFrame:
    rows = []
    mode_dir = ensure_dir(os.path.join(results_dir, "subject_specific", muscle))

    for subject in np.unique(data.subjects):
        subject_mask = data.subjects == subject
        x_subject = data.X_windows[subject_mask]
        y_subject = data.y[subject_mask]
        trial_ids_subject = data.trial_ids[subject_mask]
        times_subject = data.window_times[subject_mask]
        if len(x_subject) < 10:
            continue

        train_mask, val_mask, test_mask = choose_subject_specific_masks(trial_ids_subject)
        if train_mask.sum() < 2 or val_mask.sum() < 1 or test_mask.sum() < 1:
            continue

        x_train, y_train = x_subject[train_mask], y_subject[train_mask]
        x_val, y_val = x_subject[val_mask], y_subject[val_mask]
        x_test, y_test = x_subject[test_mask], y_subject[test_mask]
        times_test = times_subject[test_mask]
        trial_ids_test = trial_ids_subject[test_mask]

        x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, _, y_scaler = scale_sequence_windows(
            x_train, y_train, x_val, y_val, x_test
        )
        cnn_dir = ensure_dir(os.path.join(mode_dir, f"subject_{int(subject)}", "cnn_lstm"))
        model, model_meta = fit_cnn_lstm(
            x_train_s, y_train_s, x_val_s, y_val_s, config.cnn_lstm_config, cnn_dir, config.seed
        )
        pred_s = model.predict(x_test_s, verbose=0).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_s).reshape(-1, 1)
        metrics = compute_metrics(y_test, pred)
        save_prediction_artifact(
            y_test,
            pred,
            times_test,
            cnn_dir,
            f"{muscle} | CNN-LSTM | subject {int(subject)}",
            trial_ids_test,
        )
        rows.append(
            {
                "evaluation_mode": "subject_specific",
                "model": "cnn_lstm",
                "muscle": muscle,
                "test_subject": int(subject),
                "val_subject": int(subject),
                "n_train": int(len(x_train)),
                "n_val": int(len(x_val)),
                "n_test": int(len(x_test)),
                **metrics,
                **{f"meta_{k}": v for k, v in model_meta.items()},
            }
        )
    return pd.DataFrame(rows)


def run_cross_subject_xgboost_for_muscle(
    data: WindowDataset, muscle: str, splits: Sequence[Dict[str, object]], config: ExperimentConfig
) -> pd.DataFrame:
    rows = []
    for split in splits:
        train_mask = np.isin(data.subjects, split["train"])
        val_mask = np.isin(data.subjects, split["val"])
        test_mask = np.isin(data.subjects, split["test"])

        x_train, y_train = data.X_features[train_mask], data.y[train_mask]
        x_val, y_val = data.X_features[val_mask], data.y[val_mask]
        x_test, y_test = data.X_features[test_mask], data.y[test_mask]
        if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
            continue

        x_train_s, y_train_s, x_val_s, y_val_s, x_test_s, _, _ = scale_feature_matrix(
            x_train, y_train, x_val, y_val, x_test
        )
        best_model, meta = fit_best_xgboost(
            x_train_s, y_train, x_val_s, y_val, config.xgb_grid, config.seed
        )
        val_pred = best_model.predict(x_val_s).reshape(-1, 1)
        test_pred = best_model.predict(x_test_s).reshape(-1, 1)
        val_metrics = compute_metrics(y_val, val_pred)
        test_metrics = compute_metrics(y_test, test_pred)
        rows.append(
            {
                "muscle": muscle,
                "split": split["name"],
                "train_subjects": str(split["train"]),
                "val_subject": split["val"][0],
                "test_subject": split["test"][0],
                "val_r2": val_metrics["r2"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_corr": val_metrics["corr"],
                "test_r2": test_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_corr": test_metrics["corr"],
                **{f"meta_{k}": v for k, v in meta.items()},
            }
        )
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["evaluation_mode", "model", "muscle"], as_index=False)
        .agg(
            folds=("test_subject", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            corr_mean=("corr", "mean"),
            corr_std=("corr", "std"),
        )
        .sort_values(["evaluation_mode", "muscle", "r2_mean"], ascending=[True, True, False])
    )


def load_condition_data(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    emg_df = load_emg_data(config.emg_folder, config.conditions)
    suit_df = load_suit_data(config.suit_folder, config.conditions, config.suit_features)
    return emg_df, suit_df


def run_subject_specific_experiment(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
    run_name = build_run_name(config)
    results_dir = ensure_dir(os.path.join(config.results_root, run_name))
    emg_df, suit_df = load_condition_data(config)

    all_rows = []
    for muscle in config.target_muscles:
        data = build_window_dataset(
            emg_df=emg_df,
            suit_df=suit_df,
            muscle=muscle,
            suit_features=config.suit_features,
            fs=config.target_fs,
            window_seconds=config.window_seconds,
            overlap=config.window_overlap,
            target_aggregation=config.target_aggregation,
            min_trial_samples=config.min_trial_samples,
        )
        if config.run_handcrafted_models:
            fold_df = run_subject_specific_handcrafted_for_muscle(data, muscle, results_dir, config)
            if not fold_df.empty:
                all_rows.append(fold_df)
        if config.run_cnn_lstm and muscle in config.cnn_lstm_target_muscles:
            fold_df = run_subject_specific_cnn_lstm_for_muscle(data, muscle, results_dir, config)
            if not fold_df.empty:
                all_rows.append(fold_df)

    fold_results = pd.concat(all_rows, ignore_index=True)
    summary = summarize_results(fold_results)
    fold_results.to_csv(os.path.join(results_dir, f"{run_name}_fold_results.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, f"{run_name}_summary.csv"), index=False)
    return fold_results, summary


def run_cross_subject_xgboost_experiment(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
    emg_df, suit_df = load_condition_data(config)

    cross_rows = []
    for muscle in config.target_muscles:
        data = build_window_dataset(
            emg_df=emg_df,
            suit_df=suit_df,
            muscle=muscle,
            suit_features=config.suit_features,
            fs=config.target_fs,
            window_seconds=config.window_seconds,
            overlap=config.window_overlap,
            target_aggregation=config.target_aggregation,
            min_trial_samples=config.min_trial_samples,
        )
        df_muscle = run_cross_subject_xgboost_for_muscle(
            data=data,
            muscle=muscle,
            splits=config.cross_subject_splits,
            config=config,
        )
        cross_rows.append(df_muscle)

    cross_df = pd.concat(cross_rows, ignore_index=True)
    cross_summary = (
        cross_df.groupby("muscle", as_index=False)
        .agg(
            val_r2_mean=("val_r2", "mean"),
            test_r2_mean=("test_r2", "mean"),
            val_rmse_mean=("val_rmse", "mean"),
            test_rmse_mean=("test_rmse", "mean"),
            val_corr_mean=("val_corr", "mean"),
            test_corr_mean=("test_corr", "mean"),
        )
        .sort_values("muscle")
    )
    return cross_df, cross_summary
