# paper version of the LightGBM importance plot: top 10 features per target, readable names
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

from data import FEATURES, load_splits
from eval import TARGETS

ROOT = Path(__file__).resolve().parent
train = load_splits()["train"]

# settings chosen in baselines.ipynb: (num_leaves, learning_rate, trees)
PARAMS = {"hits": (4, 0.02, 662), "strikeouts": (15, 0.05, 152), "walks": (15, 0.05, 54), "earned_runs": (15, 0.05, 123)}
TITLES = {"hits": "Hits", "strikeouts": "Strikeouts", "walks": "Walks", "earned_runs": "Earned runs"}
NAMES = {
    "p_l5_h": "Last-5 hits", "p_l5_k": "Last-5 strikeouts", "p_l5_bb": "Last-5 walks", "p_l5_er": "Last-5 earned runs",
    "p_l5_outs": "Last-5 outs", "p_l5_pitches": "Last-5 pitch count", "p_rest_days": "Rest days",
    "p_n_starts": "Starts, past year", "p_k_rate": "K rate, past year", "p_bb_rate": "BB rate, past year",
    "p_h_rate": "Hit rate, past year", "p_hr_rate": "HR rate, past year", "p_er9": "ER/9, past year",
    "p_outs_per_start": "Outs per start", "p_std_n": "Starts this season", "p_std_h": "Season avg hits",
    "p_std_k": "Season avg strikeouts", "p_std_bb": "Season avg walks", "p_std_er": "Season avg earned runs",
    "p_throws_R": "Throws right", "home": "Home", "lu_pa": "Lineup PA", "lu_k_rate": "Lineup K rate",
    "lu_bb_rate": "Lineup BB rate", "lu_obp": "Lineup OBP", "lu_iso": "Lineup ISO",
    "lu_platoon_adv": "Lineup platoon adv.", "lu_bats_L": "Lineup share lefty",
}
COLOR, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 11, "font.family": "serif"})
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
for ax, t in zip(axes, TARGETS):
    leaves, lr, trees = PARAMS[t]
    m = lgb.LGBMRegressor(n_estimators=trees, num_leaves=leaves, learning_rate=lr, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    m.fit(train[FEATURES], train[t])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
    imp = (100 * imp / imp.sum()).sort_values().tail(10)
    ax.barh([NAMES[c] for c in imp.index], imp.values, height=0.6, color=COLOR)
    for y, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.02, y, f"{v:.0f}%", va="center", fontsize=9.5, color=MUTED)
    ax.set_title(TITLES[t], color=INK, fontsize=12)
    ax.set_xlim(0, imp.max() * 1.22)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, colors=INK)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3")
fig.tight_layout(w_pad=1.5)
fig.savefig(ROOT / "feature_importance_paper.png", dpi=300)
print("saved")
