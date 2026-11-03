from pathlib import Path

import pandas as pd

from eval import TARGETS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "starts_dataset.csv.gz"

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
VAL_SEASON = 2025
TEST_SEASON = 2026

SHRINK = 100  # same pseudo-count the dataset uses for other rates
BATTER_FEATS = ["pa", "k_rate", "bb_rate", "obp", "iso", "platoon_adv", "bats_L"]
BVP_BATTER_FEATS = ["bvp_ab", "bvp_avg_shrunk"]


def has_bvp(df):
    return "lu_bvp_ab" in df.columns


def add_bvp_shrunk(df):
    # raw bvp avg is NaN with no history and noisy with a few ABs, so pull it toward
    # the league bvp average from training seasons
    train = df[df.season.isin(TRAIN_SEASONS)]
    prior = train["lu_bvp_h"].sum() / train["lu_bvp_ab"].sum()
    new = {f"{b}_avg_shrunk": (df[f"{b}_h"] + prior * SHRINK) / (df[f"{b}_ab"] + SHRINK)
           for b in ["lu_bvp"] + [f"b{slot}_bvp" for slot in range(1, 10)]}
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def load(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    return add_bvp_shrunk(df) if has_bvp(df) else df


def feature_columns(df):
    # same matrix for every model: pitcher form + lineup averages + home (no bvp)
    return [c for c in df.columns if c.startswith(("p_", "lu_")) and "bvp" not in c] + ["home"]


def bvp_columns(df):
    return ["lu_bvp_ab", "lu_bvp_batters", "lu_bvp_avg_shrunk"] if has_bvp(df) else []


def batter_columns(include_bvp=False):
    # per-slot batter features for the lineup-aware model, shape (9, n_feats)
    feats = BATTER_FEATS + (BVP_BATTER_FEATS if include_bvp else [])
    return [[f"b{slot}_{f}" for f in feats] for slot in range(1, 10)]


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


_header = pd.read_csv(DATA_PATH, nrows=0)
FEATURES = feature_columns(_header)
BVP_FEATURES = bvp_columns(_header)
