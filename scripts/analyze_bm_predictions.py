"""Analyze bulk modulus predictions and diagnose energy_mult_natoms issue."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.home() / "projects" / "alignn"
ORIG = Path.home() / "projects" / "2d-alignn"

def load_metrics(name):
    path = ROOT / "results" / "logs" / f"{name}_test_metrics.json"
    with open(path) as f:
        return json.load(f)["test_metrics"]

def load_preds(name):
    return pd.read_csv(ROOT / "results" / "logs" / f"{name}_test_predictions.csv")

runs = {
    "bm_match_orig_h64_l1_e60_v2": "energy_mult_natoms + penalty (h64 L1)",
    "bm_h128_l1_e60_v2": "energy_mult_natoms + penalty (h128 L1)",
    "bm_h128_l1_e60_no_mult": "penalty only, no natoms (h128 L1)",
    "bm_h128_l1_e60_plain": "plain L1 (h128)",
}

print("=" * 70)
print("BULK MODULUS PREDICTION ANALYSIS")
print("=" * 70)

for name, desc in runs.items():
    try:
        m = load_metrics(name)
    except FileNotFoundError:
        print(f"\n--- {desc} ({name}) --- MISSING")
        continue
    preds = load_preds(name)
    errors = preds["prediction"] - preds["target"]
    abs_errors = errors.abs()

    print(f"\n--- {desc} ({name}) ---")
    print(f"  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  P95={m.get('p95_abs_error', abs_errors.quantile(0.95)):.4f}")
    print(f"  Target: min={preds['target'].min():.2f}  max={preds['target'].max():.2f}  mean={preds['target'].mean():.2f}")
    print(f"  Pred:   min={preds['prediction'].min():.2f}  max={preds['prediction'].max():.2f}  mean={preds['prediction'].mean():.2f}")
    print(f"  Bias: {errors.mean():.4f}")

    # Tail analysis
    print(f"  Tail analysis:")
    bins = [(-30, 0), (0, 50), (50, 100), (100, 200), (200, 500)]
    for lo, hi in bins:
        mask = (preds["target"] >= lo) & (preds["target"] < hi)
        n = mask.sum()
        if n > 0:
            mae = abs_errors[mask].mean()
            bias = errors[mask].mean()
            print(f"    [{lo:4d},{hi:4d}): n={n:4d}  MAE={mae:7.2f}  bias={bias:+8.2f}")

    # Top 10 worst errors
    preds["abs_error"] = abs_errors
    worst = preds.nlargest(10, "abs_error")
    print(f"  Top 10 worst predictions:")
    for _, row in worst.iterrows():
        print(f"    target={row['target']:7.2f}  pred={row['prediction']:7.2f}  err={row['prediction']-row['target']:+8.2f}")

# Original comparison
print(f"\n--- Original (e60) ---")
try:
    orig_results = pd.read_csv(ORIG / "fair_compare_dft3d_bulk_modulus_kv_atomwise_e60" / "prediction_results_test_set.csv")
    print(f"  Columns: {list(orig_results.columns)}")
    print(orig_results.head())
except Exception as e:
    print(f"  Error loading: {e}")
    try:
        with open(ORIG / "fair_compare_dft3d_bulk_modulus_kv_atomwise_e60" / "Test_results.json") as f:
            orig = json.load(f)
        print(f"  {orig}")
    except Exception as e2:
        print(f"  Error: {e2}")

# Compare plain vs no_mult
print(f"\n--- Comparison: plain vs no_mult ---")
p_plain = load_preds("bm_h128_l1_e60_plain")
p_nomult = load_preds("bm_h128_l1_e60_no_mult")
# Check if they use the same test JIDs
if "jid" in p_plain.columns and "jid" in p_nomult.columns:
    merged = p_plain.merge(p_nomult, on="jid", suffixes=("_plain", "_nomult"))
    print(f"  Matched samples: {len(merged)}")
    plain_better = (merged["abs_error" if "abs_error" in merged else "prediction_plain"].abs() < merged["prediction_nomult"].abs()).sum()
else:
    print(f"  JID columns: plain={p_plain.columns.tolist()[:5]}, nomult={p_nomult.columns.tolist()[:5]}")
