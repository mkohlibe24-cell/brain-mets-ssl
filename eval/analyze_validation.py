"""
Analyzes /workspace/eval/validation_results.csv (output of validate_vs_gt.py)
to answer the two gate questions before training:
  1. Which threshold (0.85 / 0.90 / 0.95) gives the best Dice vs BraTS GT?
  2. Is the "partial-inversion" (partial-input KEEP% > full-input KEEP%) a
     bucketing artifact, or do partial-input predictions genuinely score
     WORSE Dice vs GT (meaning: down-weight/drop partial pseudo-labels)?

Also reports a "total miss" rate (dice == 0.0) per mode/threshold as a proxy
for false-negative rate, since validate_vs_gt.py does not currently emit
voxel-level FN counts.

Usage (run inside the container, after validate_vs_gt.py finishes or
partway through — this script is safe to run on a partial CSV too):
    /root/aurora_env/bin/python3 /workspace/eval/analyze_validation.py

Output: prints summary tables to stdout AND writes
    /workspace/eval/validation_summary.csv
    /workspace/eval/threshold_recommendation.txt
"""
import sys
from pathlib import Path

import pandas as pd

IN_CSV = Path('/workspace/eval/validation_results.csv')
OUT_SUMMARY_CSV = Path('/workspace/eval/validation_summary.csv')
OUT_RECOMMENDATION_TXT = Path('/workspace/eval/threshold_recommendation.txt')

MODE_LABELS = {
    'full': 'full (t1/t1c/t2/fla)',
    'partial_t1c_t1n': 'partial (t1c-t1n)',
    'partial_t1c_fla': 'partial (t1c-fla)',
}


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: {path} does not exist yet. Is validate_vs_gt.py running?")
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"ERROR: {path} exists but has 0 rows yet. Wait for more progress.")
    df['threshold'] = df['threshold'].astype(float)
    df['dice'] = df['dice'].astype(float)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in df['mode'].unique():
        for thr in sorted(df['threshold'].unique()):
            sub = df[(df['mode'] == mode) & (df['threshold'] == thr)]
            if sub.empty:
                continue
            total_miss = (sub['dice'] == 0.0).sum()
            rows.append({
                'mode': mode,
                'threshold': thr,
                'n_cases': len(sub),
                'dice_mean': sub['dice'].mean(),
                'dice_median': sub['dice'].median(),
                'dice_std': sub['dice'].std(),
                'total_miss_rate': total_miss / len(sub),
            })
    return pd.DataFrame(rows).sort_values(['mode', 'threshold'])


def print_table(summary: pd.DataFrame):
    print("\n=== Dice by mode x threshold ===")
    for mode in summary['mode'].unique():
        sub = summary[summary['mode'] == mode]
        label = MODE_LABELS.get(mode, mode)
        print(f"\n{label}  (n={int(sub['n_cases'].iloc[0])} cases)")
        print(f"{'thr':>6} {'mean':>8} {'median':>8} {'std':>8} {'miss_rate':>10}")
        for _, r in sub.iterrows():
            print(f"{r['threshold']:>6.2f} {r['dice_mean']:>8.4f} {r['dice_median']:>8.4f} "
                  f"{r['dice_std']:>8.4f} {r['total_miss_rate']:>10.2%}")


def test_partial_inversion(summary: pd.DataFrame):
    print("\n=== Partial-inversion check (Dice vs GT, not KEEP%) ===")
    print("Hypothesis from bucketing: partial-input cases survived KEEP more often")
    print("(74% vs 56% @0.85), suspected to mean blobbier/over-confident, LESS accurate.")
    print("This checks it directly against ground truth.\n")

    verdict_lines = []
    for thr in sorted(summary['threshold'].unique()):
        full_row = summary[(summary['mode'] == 'full') & (summary['threshold'] == thr)]
        part_rows = summary[(summary['mode'] != 'full') & (summary['threshold'] == thr)]
        if full_row.empty or part_rows.empty:
            continue
        full_dice = full_row['dice_mean'].iloc[0]
        for _, prow in part_rows.iterrows():
            gap = full_dice - prow['dice_mean']
            direction = "WORSE than full" if gap > 0.01 else (
                "comparable to full" if abs(gap) <= 0.01 else "BETTER than full (unexpected)")
            line = (f"thr={thr:.2f}: {MODE_LABELS.get(prow['mode'], prow['mode'])} "
                    f"mean Dice={prow['dice_mean']:.4f} vs full={full_dice:.4f} "
                    f"(gap={gap:+.4f}) -> {direction}")
            print(line)
            verdict_lines.append(line)
    return verdict_lines


def recommend(summary: pd.DataFrame) -> str:
    full = summary[summary['mode'] == 'full']
    if full.empty:
        return "Not enough 'full' mode data yet to recommend a threshold."

    best_thr_row = full.loc[full['dice_mean'].idxmax()]
    best_thr = best_thr_row['threshold']

    partial_modes = summary[summary['mode'] != 'full']
    partial_worse_everywhere = True
    if not partial_modes.empty:
        for thr in sorted(summary['threshold'].unique()):
            full_d = summary[(summary['mode'] == 'full') & (summary['threshold'] == thr)]
            part_d = summary[(summary['mode'] != 'full') & (summary['threshold'] == thr)]
            if full_d.empty or part_d.empty:
                continue
            if part_d['dice_mean'].max() >= full_d['dice_mean'].iloc[0] - 0.01:
                partial_worse_everywhere = False

    lines = [
        f"RECOMMENDED THRESHOLD (by full-input Dice): {best_thr:.2f}",
        f"  full-input mean Dice at this threshold: {best_thr_row['dice_mean']:.4f}",
        f"  full-input total-miss rate at this threshold: {best_thr_row['total_miss_rate']:.2%}",
        "",
    ]
    if partial_worse_everywhere:
        lines.append(
            "PARTIAL-INPUT POLICY: partial-sequence Dice is consistently lower than "
            "full-sequence Dice across thresholds. This CONFIRMS the inversion hypothesis: "
            "high partial KEEP% in bucketing reflected over-confident/blobbier predictions, "
            "not more accurate ones. Recommendation: down-weight partial-sequence pseudo-labels "
            "in training (e.g. lower sample weight, or drop entirely) despite their high KEEP%."
        )
    else:
        lines.append(
            "PARTIAL-INPUT POLICY: partial-sequence Dice is NOT consistently lower than "
            "full-sequence Dice. The inversion may be more benign than assumed -- inspect the "
            "per-threshold gaps above before deciding whether to down-weight partials."
        )
    return "\n".join(lines)


def main():
    df = load(IN_CSV)
    print(f"Loaded {len(df)} rows from {IN_CSV}")
    print(f"Unique cases seen so far: {df['case_id'].nunique()}")
    print(f"Modes present: {sorted(df['mode'].unique())}")

    summary = summarize(df)
    print_table(summary)
    test_partial_inversion(summary)

    rec = recommend(summary)
    print("\n=== RECOMMENDATION ===")
    print(rec)

    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    OUT_RECOMMENDATION_TXT.write_text(rec + "\n")
    print(f"\nWrote {OUT_SUMMARY_CSV}")
    print(f"Wrote {OUT_RECOMMENDATION_TXT}")


if __name__ == '__main__':
    main()
