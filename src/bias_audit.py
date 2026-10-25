"""Subgroup audit used in the report (Section 4).

Run from the project root:  python src/bias_audit.py
Paste into notebook 02 (new Section 7.4) if you prefer it inline.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_1000"
PRED = ROOT / "outputs" / "results_v2" / "test_predictions.csv"
META = ROOT / "data" / "interim" / "final_clean_metadata.csv"
GREY_THRESHOLD = 3.0  # mean inter-channel difference on a 0-255 scale


def chroma(path):
    im = Image.open(path)
    im.draft("RGB", (200, 150))
    a = np.asarray(im.convert("RGB").resize((160, 120)), dtype=np.int16)[:110]  # drop info bar
    return np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()


def rates(df, group):
    alert, no_alert = df[df.true_alert == 1], df[df.true_alert == 0]
    return pd.DataFrame({
        "alert_recall": alert.groupby(group).apply(lambda g: (g.predicted_alert == 1).mean(), include_groups=False),
        "n_alert": alert.groupby(group).size(),
        "false_alert_rate": no_alert.groupby(group).apply(lambda g: (g.predicted_alert == 1).mean(), include_groups=False),
        "n_no_alert": no_alert.groupby(group).size(),
    }).round(4)


test = pd.read_csv(PRED)
test["infrared"] = [chroma(CACHE / "test" / r.project_class / r.file) < GREY_THRESHOLD for r in test.itertuples()]
hour = pd.to_datetime(test["date"], format="mixed", errors="coerce").dt.hour
test["night"] = hour.isin(list(range(0, 6)) + list(range(19, 24)))

print("By image mode (infrared = greyscale):\n", rates(test, "infrared"), "\n")
print("By capture time (night = 19:00-05:59):\n", rates(test, "night"), "\n")

train = pd.read_csv(META).query("split == 'train'").groupby("project_class").sample(300, random_state=0)
train["infrared"] = [chroma(CACHE / "train" / r.project_class / r.file) < GREY_THRESHOLD for r in train.itertuples()]
print("Share of greyscale infrared frames in training sample (%):\n",
      (train.groupby("project_class").infrared.mean() * 100).round(1), "\n")

correct = test.assign(correct=test.predicted_class == test.true_class)
site = correct.groupby("site").agg(n=("file", "size"), accuracy=("correct", "mean"))
print("Per-site accuracy (sites with >= 30 test images):\n", site[site.n >= 30].accuracy.describe().round(3))
