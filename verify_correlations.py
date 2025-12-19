#!/usr/bin/env python3
"""
Compute correlations from original v3_rebalanced dataset
"""
import json
import numpy as np

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# Load data
data = load_jsonl("/home/seung/mmhoa/text2hoa/v2/text2spatial_v3_rebalanced.jsonl")
print(f"Loaded {len(data)} samples")

# Extract parameters
distances = []
gains = []
wets = []
spreads = []

for item in data:
    dist = float(item.get('dist_m', 1.0))
    gain = float(item.get('gain_db', -3.0))
    wet = float(item.get('wet_mix', 0.2))
    spread = float(item.get('spread_deg', 30.0))

    distances.append(dist)
    gains.append(gain)
    wets.append(wet)
    spreads.append(spread)

distances = np.array(distances)
gains = np.array(gains)
wets = np.array(wets)
spreads = np.array(spreads)

# Compute correlations
corr_dist_gain = np.corrcoef(distances, gains)[0, 1]
corr_dist_wet = np.corrcoef(distances, wets)[0, 1]
corr_dist_spread = np.corrcoef(distances, spreads)[0, 1]

print("\n" + "="*60)
print("CORRELATION ANALYSIS (Full Dataset)")
print("="*60)
print(f"\nDataset: {len(data)} samples")
print(f"\nParameter ranges:")
print(f"  Distance: {np.min(distances):.2f} - {np.max(distances):.2f} m (mean: {np.mean(distances):.2f})")
print(f"  Gain:     {np.min(gains):.2f} - {np.max(gains):.2f} dB (mean: {np.mean(gains):.2f})")
print(f"  Wet:      {np.min(wets):.2f} - {np.max(wets):.2f} (mean: {np.mean(wets):.2f})")
print(f"  Spread:   {np.min(spreads):.2f} - {np.max(spreads):.2f}° (mean: {np.mean(spreads):.2f})")

print(f"\n📊 CORRELATIONS:")
print(f"  distance ↔ gain:   ρ = {corr_dist_gain:+.3f}")
print(f"  distance ↔ wet:    ρ = {corr_dist_wet:+.3f}")
print(f"  distance ↔ spread: ρ = {corr_dist_spread:+.3f}")

print(f"\n📝 REBUTTAL CLAIMS:")
print(f"  Claimed: distance↔gain (ρ=-0.23)")
print(f"  Actual:  distance↔gain (ρ={corr_dist_gain:.2f}) ", end="")
if abs(corr_dist_gain - (-0.23)) < 0.05:
    print("✅ VERIFIED (within ±0.05)")
else:
    print(f"⚠️  DIFFERS by {abs(corr_dist_gain - (-0.23)):.3f}")
    print(f"     → Update rebuttal to: ρ={corr_dist_gain:.2f}")

print(f"\n  Claimed: distance↔wet (ρ=0.15)")
print(f"  Actual:  distance↔wet (ρ={corr_dist_wet:.2f}) ", end="")
if abs(corr_dist_wet - 0.15) < 0.05:
    print("✅ VERIFIED (within ±0.05)")
else:
    print(f"⚠️  DIFFERS by {abs(corr_dist_wet - 0.15):.3f}")
    print(f"     → Update rebuttal to: ρ={corr_dist_wet:.2f}")

print(f"\n  Claimed: distance↔spread (ρ=0.08)")
print(f"  Actual:  distance↔spread (ρ={corr_dist_spread:.2f}) ", end="")
if abs(corr_dist_spread - 0.08) < 0.05:
    print("✅ VERIFIED (within ±0.05)")
else:
    print(f"⚠️  DIFFERS by {abs(corr_dist_spread - 0.08):.3f}")
    print(f"     → Update rebuttal to: ρ={corr_dist_spread:.2f}")

print(f"\n🔍 STATISTICAL INTERPRETATION:")
for name, rho in [("distance↔gain", corr_dist_gain),
                   ("distance↔wet", corr_dist_wet),
                   ("distance↔spread", corr_dist_spread)]:
    abs_rho = abs(rho)
    if abs_rho < 0.1:
        strength = "NEGLIGIBLE"
        desc = "very weak"
    elif abs_rho < 0.3:
        strength = "WEAK"
        desc = "weak to moderate"
    elif abs_rho < 0.5:
        strength = "MODERATE"
        desc = "moderate"
    elif abs_rho < 0.7:
        strength = "STRONG"
        desc = "strong"
    else:
        strength = "VERY STRONG"
        desc = "very strong"
    print(f"  {name:20s}: |ρ|={abs_rho:.3f} → {strength:12s} ({desc})")

print(f"\n✅ RECOMMENDATION:")
if all(abs(r) < 0.3 for r in [corr_dist_gain, corr_dist_wet, corr_dist_spread]):
    print("  ✓ All correlations are WEAK (|ρ| < 0.3)")
    print("  ✓ Can claim: 'These weak-to-moderate correlations confirm distinct semantic learning'")
    print("  ✓ Better wording: 'moderate correlations' (safer than 'low')")
else:
    print("  ⚠️  Some correlations exceed 0.3")
    print("  → Must use 'moderate' instead of 'low'")

print(f"\n🎯 UPDATED REBUTTAL TEXT:")
print(f"  **Section 4.1.1** explains perceptual independence despite physical coupling.")
print(f"  Correlation analysis: distance↔gain (ρ={corr_dist_gain:.2f}), distance↔wet")
print(f"  (ρ={corr_dist_wet:.2f}), distance↔spread (ρ={corr_dist_spread:.2f}). These")
print(f"  {'weak-to-moderate' if all(abs(r)<0.3 for r in [corr_dist_gain,corr_dist_wet,corr_dist_spread]) else 'moderate'}")
print(f"  correlations confirm distinct semantic learning, enabling creative control")
print(f"  like 'whisper from far away.'")

print("\n" + "="*60)
