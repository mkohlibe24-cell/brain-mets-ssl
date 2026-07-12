"""
Decides, for each Yale case in the EMPTY bucket at threshold 0.85, whether it
looks like a TRUE NEGATIVE (genuinely no mets -> valid training example with an
empty label) or a LIKELY AURORA MISS (small/faint lesion present but under the
KEEP threshold -> should be excluded from training rather than taught as a hard
negative).

MANIFEST FORMAT NOTE: bucket_manifest.csv is WIDE-format -- one row per case,
with a separate bucket column per threshold ('bucket@0.95', 'bucket@0.9',
'bucket@0.85'), not a single 'bucket' + 'threshold' pair of columns. An earlier
version of this script auto-detected the bucket column with a generic
'bucket' in c.lower() match, which grabbed whichever bucket@ column happened
to come first (bucket@0.95) regardless of THRESHOLD, and silently skipped
filtering because no 'threshold' column exists in this format. That silently
computed the EMPTY-bucket policy on the wrong threshold's cases. Fixed by
explicitly building the column name f"bucket@{THRESHOLD}" and failing loudly
if it isn't found instead of falling back to a guess.

CALIBRATION NOTE (updated after manifest-column fix): with the CORRECT
490-case EMPTY population at thr=0.85 (see manifest note above -- an earlier
run had wrongly swept the 554-case thr=0.95 EMPTY population and never got
close to the BraTS reference rate), the sweep now lands almost exactly on
target: LOW_CONF_THRESH=0.70 -> LIKELY_MISS fraction = 10.4%, vs the
BRATS_MISS_RATE reference of 10.5%. This is a genuine calibration match, not
a coincidence of a miscalibrated population, so LOW_CONF_THRESH_DEFAULT is
set to 0.70 (51 LIKELY_MISS / 439 TRUE_NEGATIVE), superseding the earlier
conservative 0.3 default (356/198 on the wrong population) that was only
adopted because no threshold fit the target on the wrong 554-case set.
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

BUCKET_MANIFEST = Path('/workspace/pseudolabels/buckets/bucket_manifest.csv')
PROB_MAPS_DIR = Path('/workspace/pseudolabels/prob_maps')
OUT_CSV = Path('/workspace/pseudolabels/buckets/empty_policy_thr0.85.csv')

THRESHOLD = 0.85
BRATS_MISS_RATE = 0.105
LOW_CONF_THRESH_DEFAULT = 0.70
SWEEP_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_manifest():
    if not BUCKET_MANIFEST.exists():
        sys.exit(f"ERROR: {BUCKET_MANIFEST} not found.")
    df = pd.read_csv(BUCKET_MANIFEST)
    print(f"Loaded bucket manifest: {len(df)} rows, columns: {list(df.columns)}")

    case_col = next((c for c in df.columns if c.lower() in ('case_id', 'caseid', 'case')), None)
    if case_col is None:
        sys.exit(f"ERROR: could not auto-detect case_id column. Found: {list(df.columns)}")

    # Wide-format manifest: one bucket column per threshold, e.g. 'bucket@0.85'.
    # Build the exact expected column name first; only fall back to a tolerant
    # float-match search (to handle '0.9' vs '0.90' style formatting) if the
    # exact name isn't present. Never silently pick an unrelated bucket@ column.
    bucket_col = f"bucket@{THRESHOLD}"
    if bucket_col not in df.columns:
        candidates = [
            c for c in df.columns
            if c.startswith('bucket@') and float(c.split('@')[1]) == THRESHOLD
        ]
        if not candidates:
            sys.exit(
                f"ERROR: no bucket column found for threshold {THRESHOLD}. "
                f"Available columns: {list(df.columns)}"
            )
        bucket_col = candidates[0]

    print(f"Using case_id='{case_col}', bucket_col='{bucket_col}'")

    empty_df = df[df[bucket_col].astype(str).str.upper() == 'EMPTY'].copy()
    empty_df = empty_df.rename(columns={case_col: 'case_id'})
    print(f"EMPTY-bucket cases at thr={THRESHOLD}: {len(empty_df)}")
    return empty_df[['case_id']].drop_duplicates()


def find_prob_map(case_id):
    matches = glob.glob(str(PROB_MAPS_DIR / f"*{case_id}*"))
    matches = [m for m in matches if m.endswith('.nii.gz') or m.endswith('.nii')]
    return matches[0] if matches else None


def compute_max_probs(empty_df):
    rows = []
    n_missing = 0
    for i, case_id in enumerate(empty_df['case_id']):
        path = find_prob_map(case_id)
        if path is None:
            n_missing += 1
            continue
        try:
            arr = nib.load(path).get_fdata()
            rows.append({'case_id': case_id, 'max_prob': float(np.max(arr))})
        except Exception as e:
            print(f"[{case_id}] failed: {e}")
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(empty_df)}")
    if n_missing:
        print(f"WARNING: {n_missing} cases had no matching prob map")
    return pd.DataFrame(rows)


def sweep_thresholds(df):
    print(f"\n=== Calibration sweep (target ~{BRATS_MISS_RATE:.1%}) ===")
    for t in SWEEP_VALUES:
        frac = (df['max_prob'] >= t).mean()
        flag = "  <-- closest" if abs(frac - BRATS_MISS_RATE) < 0.03 else ""
        print(f"LOW_CONF_THRESH={t:.2f}: LIKELY_MISS fraction = {frac:.1%}{flag}")


def main():
    empty_df = load_manifest()
    if empty_df.empty:
        sys.exit("No EMPTY-bucket cases found.")

    print(f"\nComputing max probability per EMPTY case from {PROB_MAPS_DIR} ...")
    prob_df = compute_max_probs(empty_df)
    if prob_df.empty:
        sys.exit("Could not load any prob maps.")

    sweep_thresholds(prob_df)

    prob_df['decision'] = np.where(
        prob_df['max_prob'] >= LOW_CONF_THRESH_DEFAULT, 'LIKELY_MISS', 'TRUE_NEGATIVE'
    )
    n_true_neg = (prob_df['decision'] == 'TRUE_NEGATIVE').sum()
    n_likely_miss = (prob_df['decision'] == 'LIKELY_MISS').sum()

    print(f"\n=== Using LOW_CONF_THRESH={LOW_CONF_THRESH_DEFAULT} ===")
    print(f"TRUE_NEGATIVE: {n_true_neg}")
    print(f"LIKELY_MISS:   {n_likely_miss}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    prob_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == '__main__':
    main()