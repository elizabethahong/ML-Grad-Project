import numpy as np
import pandas as pd

# rename to match the dataset columns
TARGETS = ["hits", "strikeouts", "walks", "earned_runs"]


def _to_array(y, targets):
    if isinstance(y, pd.DataFrame):
        y = y[targets]
    if hasattr(y, "detach"):  # torch tensor
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=float)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def evaluate(y_true, y_pred, targets=TARGETS):
    # returns MAE and RMSE for each target
    targets = list(targets)
    yt = _to_array(y_true, targets)
    yp = _to_array(y_pred, targets)

    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.shape[1] != len(targets):
        raise ValueError(f"expected {len(targets)} columns, got {yt.shape[1]}")
    if np.isnan(yt).any() or np.isnan(yp).any():
        raise ValueError("NaNs in y_true or y_pred")

    err = yp - yt
    return pd.DataFrame(
        {"MAE": np.abs(err).mean(axis=0), "RMSE": np.sqrt((err**2).mean(axis=0))},
        index=pd.Index(targets, name="target"),
    )


def compare(results):
    # {model_name: evaluate(...)} -> one row per model
    out = pd.DataFrame({name: t.stack() for name, t in results.items()}).T
    out.index.name = "model"
    return out
