#!/usr/bin/env python3
"""Compare v5b and v5d results and simulate ensemble."""
import pyarrow.parquet as pq
import numpy as np

tb = pq.read_table("results/video_20260218_014220/records.parquet").to_pandas()
td = pq.read_table("results/video_20260218_060823/records.parquet").to_pandas()

wrong_b = set(tb[tb["predicted"] != tb["label"]].index.tolist())
wrong_d = set(td[td["predicted"] != td["label"]].index.tolist())

both_wrong = wrong_b & wrong_d
only_b_wrong = wrong_b - wrong_d
only_d_wrong = wrong_d - wrong_b

print(f"v5b wrong: {len(wrong_b)}, v5d wrong: {len(wrong_d)}")
print(f"Both wrong: {len(both_wrong)}")
print(f"Only v5b wrong: {len(only_b_wrong)}")
print(f"Only v5d wrong: {len(only_d_wrong)}")

print("\nBoth wrong:")
for idx in sorted(both_wrong):
    rb = tb.iloc[idx]
    rd = td.iloc[idx]
    ds = rb["dataset_name"]
    lbl = rb["label"]
    pb = rb["probs"][1]
    pd_ = rd["probs"][1]
    print(f"  {ds}: true={lbl} v5b_P={pb:.4f} v5d_P={pd_:.4f}")

print("\nOnly v5b wrong:")
for idx in sorted(only_b_wrong):
    rb = tb.iloc[idx]
    rd = td.iloc[idx]
    ds = rb["dataset_name"]
    lbl = rb["label"]
    pb = rb["probs"][1]
    pd_ = rd["probs"][1]
    print(f"  {ds}: true={lbl} v5b_P={pb:.4f} v5d_P={pd_:.4f}")

print("\nOnly v5d wrong:")
for idx in sorted(only_d_wrong):
    rb = tb.iloc[idx]
    rd = td.iloc[idx]
    ds = rd["dataset_name"]
    lbl = rd["label"]
    pb = rb["probs"][1]
    pd_ = rd["probs"][1]
    print(f"  {ds}: true={lbl} v5b_P={pb:.4f} v5d_P={pd_:.4f}")

# Simulate ensemble
prob_b = np.array([p[1] for p in tb["probs"]])
prob_d = np.array([p[1] for p in td["probs"]])
labels = tb["label"].values

# Average probabilities
prob_ens = (prob_b + prob_d) / 2.0
pred_ens = (prob_ens > 0.5).astype(int)
wrong_ens = pred_ens != labels

brier_ens = np.mean((prob_ens - labels) ** 2)
tp = ((pred_ens==1) & (labels==1)).sum()
tn = ((pred_ens==0) & (labels==0)).sum()
fp = ((pred_ens==1) & (labels==0)).sum()
fn = ((pred_ens==0) & (labels==1)).sum()
d = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
mcc = float(tp*tn - fp*fn) / d if d > 0 else 0
mcc_norm = max(0, ((mcc+1)/2)) ** 1.2
brier_norm = max(0, (0.25-brier_ens)/0.25) ** 1.8
sn34 = np.sqrt(max(1e-12, mcc_norm * brier_norm))

print(f"\nENSEMBLE (avg probs): wrong={wrong_ens.sum()}, MCC={mcc:.4f}, Brier={brier_ens:.6f}, SN34={sn34:.4f}")

# Try different ensemble weights
print("\nWeighted ensemble sweep:")
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    prob_w = w * prob_b + (1 - w) * prob_d
    pred_w = (prob_w > 0.5).astype(int)
    brier_w = np.mean((prob_w - labels) ** 2)
    tp2 = ((pred_w==1) & (labels==1)).sum()
    tn2 = ((pred_w==0) & (labels==0)).sum()
    fp2 = ((pred_w==1) & (labels==0)).sum()
    fn2 = ((pred_w==0) & (labels==1)).sum()
    d2 = np.sqrt(float((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2)))
    mcc2 = float(tp2*tn2 - fp2*fn2) / d2 if d2 > 0 else 0
    mcc_norm2 = max(0, ((mcc2+1)/2)) ** 1.2
    brier_norm2 = max(0, (0.25-brier_w)/0.25) ** 1.8
    sn34_2 = np.sqrt(max(1e-12, mcc_norm2 * brier_norm2))
    wrong_w = (pred_w != labels).sum()
    print(f"  w_v5b={w:.1f}: wrong={wrong_w}, MCC={mcc2:.4f}, Brier={brier_w:.6f}, SN34={sn34_2:.4f}")
