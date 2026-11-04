import numpy as np
import pandas as pd

from eval import TARGETS, compare, evaluate

# true values are all 0, so errors = predictions
y_true = np.zeros((2, 4))
y_pred = np.array([[1, 0, -2, 4],
                   [3, 0,  2, 0]])

t = evaluate(y_true, y_pred)
assert list(t.index) == TARGETS
assert np.allclose(t["MAE"], [2, 0, 2, 2])
assert np.allclose(t["RMSE"], [np.sqrt(5), 0, 2, np.sqrt(8)])

# column order shouldn't matter for dataframes
df_true = pd.DataFrame(y_true, columns=TARGETS)[TARGETS[::-1]]
df_pred = pd.DataFrame(y_pred, columns=TARGETS)
assert np.allclose(evaluate(df_true, df_pred).values, t.values)

# wrong shape, NaNs, wrong number of columns
for bad in [np.zeros((3, 4)), np.full((2, 4), np.nan), np.zeros((2, 3))]:
    try:
        evaluate(y_true, bad)
        raise AssertionError
    except ValueError:
        pass

c = compare({"a": t, "b": evaluate(y_true, y_true)})
assert list(c.index) == ["a", "b"]
assert c.loc["b"].abs().sum() == 0

print(c.round(3))
print("ok")
