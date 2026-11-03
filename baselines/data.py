from pathlib import Path

import pandas as pd

from eval import TARGETS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "starts_dataset.csv.gz"

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
VAL_SEASON = 2025
TEST_SEASON = 2026

BATTER_FEATS = ["pa", "k_rate", "bb_rate", "obp", "iso", "platoon_adv", "bats_L"]


def load(path=DATA_PATH):
    return pd.read_csv(path, parse_dates=["date"])


def feature_columns(df):
    # same matrix for every model: pitcher form + lineup averages + home
    return [c for c in df.columns if c.startswith(("p_", "lu_"))] + ["home"]


def batter_columns():
    # per-slot batter features for the lineup-aware model, shape (9, len(BATTER_FEATS))
    return [[f"b{slot}_{f}" for f in BATTER_FEATS] for slot in range(1, 10)]


def load_splits(path=DATA_PATH, include_test=False):
    df = load(path)
    splits = {
        "train": df[df.season.isin(TRAIN_SEASONS)].reset_index(drop=True),
        "val": df[df.season == VAL_SEASON].reset_index(drop=True),
    }
    # 2026 stays untouched until the final eval
    if include_test:
        splits["test"] = df[df.season == TEST_SEASON].reset_index(drop=True)
    return splits


FEATURES = feature_columns(pd.read_csv(DATA_PATH, nrows=0))
