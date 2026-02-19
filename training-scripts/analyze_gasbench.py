#!/usr/bin/env python3
"""Analyze gasbench records to find optimal temperature/threshold."""
import pyarrow.parquet as pq
import numpy as np
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else '/root/bitmind-subnet/results/video_20260218_014220'

t = pq.read_table(f'{results_dir}/records.parquet')
df = t.to_pandas()

prob_synth = np.array([p[1] for p in df['probs']])
labels = df['label'].values
predicted = df['predicted'].values

# Show wrong predictions
wrong_mask = predicted != labels
print("WRONG PREDICTIONS:")
for idx in np.where(wrong_mask)[0]:
    row = df.iloc[idx]
    p = row['probs']
    conf_tag = "CONFIDENT WRONG" if (row['label']==1 and p[1]<0.2) or (row['label']==0 and p[1]>0.8) else ""
    print(f"  {row['dataset_name']}: true={row['label']} pred={row['predicted']} P(synth)={p[1]:.4f} {conf_tag}")

print(f"\nTotal wrong: {wrong_mask.sum()} / {len(df)}")

# Current metrics
brier = np.mean((prob_synth - labels) ** 2)
tp = ((predicted==1) & (labels==1)).sum()
tn = ((predicted==0) & (labels==0)).sum()
fp = ((predicted==1) & (labels==0)).sum()
fn = ((predicted==0) & (labels==1)).sum()
d = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
mcc = float(tp*tn - fp*fn) / d if d > 0 else 0
mcc_norm = max(0, ((mcc+1)/2)) ** 1.2
brier_norm = max(0, (0.25-brier)/0.25) ** 1.8
sn34 = np.sqrt(max(1e-12, mcc_norm * brier_norm))
print(f"\nCurrent: MCC={mcc:.4f} Brier={brier:.6f} sn34={sn34:.4f}")

# Threshold sweep (only affects MCC, not Brier)
print("\nThreshold sweep:")
for thr in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    new_pred = (prob_synth > thr).astype(int)
    tp2 = ((new_pred==1) & (labels==1)).sum()
    tn2 = ((new_pred==0) & (labels==0)).sum()
    fp2 = ((new_pred==1) & (labels==0)).sum()
    fn2 = ((new_pred==0) & (labels==1)).sum()
    d2 = np.sqrt(float((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2)))
    mcc2 = float(tp2*tn2 - fp2*fn2) / d2 if d2 > 0 else 0
    correct2 = (new_pred == labels).sum()
    mcc_norm2 = max(0, ((mcc2+1)/2)) ** 1.2
    brier_norm2 = max(0, (0.25-brier)/0.25) ** 1.8
    sn34_2 = np.sqrt(max(1e-12, mcc_norm2 * brier_norm2))
    print(f"  thr={thr:.2f}: correct={correct2}/{len(labels)} MCC={mcc2:.4f} sn34={sn34_2:.4f}")

# Check if scaling probabilities helps Brier
print("\nBrier with scaled probabilities:")
for scale in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
    # Scale logits (inverse of temperature scaling on probs)
    # Convert probs back to logits, scale, convert back
    eps = 1e-7
    logit = np.log(prob_synth.clip(eps, 1-eps) / (1-prob_synth).clip(eps, 1-eps))
    scaled_logit = logit * scale
    scaled_prob = 1 / (1 + np.exp(-scaled_logit))
    new_brier = np.mean((scaled_prob - labels) ** 2)
    new_pred = (scaled_prob > 0.5).astype(int)
    tp3 = ((new_pred==1) & (labels==1)).sum()
    tn3 = ((new_pred==0) & (labels==0)).sum()
    fp3 = ((new_pred==1) & (labels==0)).sum()
    fn3 = ((new_pred==0) & (labels==1)).sum()
    d3 = np.sqrt(float((tp3+fp3)*(tp3+fn3)*(tn3+fp3)*(tn3+fn3)))
    mcc3 = float(tp3*tn3 - fp3*fn3) / d3 if d3 > 0 else 0
    mcc_norm3 = max(0, ((mcc3+1)/2)) ** 1.2
    brier_norm3 = max(0, (0.25-new_brier)/0.25) ** 1.8
    sn34_3 = np.sqrt(max(1e-12, mcc_norm3 * brier_norm3))
    print(f"  scale={scale:.2f}: Brier={new_brier:.6f} MCC={mcc3:.4f} sn34={sn34_3:.4f}")
