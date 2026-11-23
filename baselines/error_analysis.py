# LightGBM errors on 2025 split by outing length, time of season and pitcher history
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import FEATURES, load_splits
from eval import TARGETS

ROOT = Path(__file__).resolve().parent.parent
splits = load_splits()
train, val = splits["train"], splits["val"]

# settings chosen in baselines.ipynb: (num_leaves, learning_rate, trees)
PARAMS = {"hits": (4, 0.02, 662), "strikeouts": (15, 0.05, 152), "walks": (15, 0.05, 54), "earned_runs": (15, 0.05, 123)}
pred = {}
for t, (leaves, lr, trees) in PARAMS.items():
    m = lgb.LGBMRegressor(n_estimators=trees, num_leaves=leaves, learning_rate=lr, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    pred[t] = m.fit(train[FEATURES], train[t]).predict(val[FEATURES])

outs = pd.read_csv(ROOT / "data" / "raw" / "starts_2025.csv.gz")[["game_pk", "pitcher_id", "outs"]]
val = val.merge(outs, on=["game_pk", "pitcher_id"], how="left")
groups = pd.DataFrame({
    "outing": pd.cut(val.outs, [-1, 14, 17, 99], labels=["< 5 IP", "5-5.2 IP", "6+ IP"]),
    "phase": pd.cut(val.date.dt.month, [0, 4, 7, 12], labels=["Mar-Apr", "May-Jul", "Aug-Sep"]),
    "history": pd.cut(val.p_n_starts, [-1, 9, 24, 999], labels=["< 10 starts", "10-24", "25+"]),
})

rows = []
for split in groups:
    for level, idx in groups.groupby(split, observed=True).groups.items():
        row = {"split": split, "group": level, "n": len(idx)}
        for t in TARGETS:
            err = pred[t][idx] - val.loc[idx, t]
            row[f"{t}_rmse"] = np.sqrt(np.mean(err ** 2))
            row[f"{t}_bias"] = err.mean()
        rows.append(row)
table = pd.DataFrame(rows).round(3)
table.to_csv(ROOT / "baselines" / "error_analysis_2025.csv", index=False)
print(table.to_string())
