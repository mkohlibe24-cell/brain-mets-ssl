"""
Assembles the final training cohort combining:
  - 200 real BraTS cases (ground-truth segmentation, use as-is)
  - Yale KEEP-bucket cases (pseudo-labeled by AURORA, continuous prob map,
    binarized at THRESHOLD at train time)
  - Yale EMPTY-bucket cases classified TRUE_NEGATIVE by empty_bucket_policy.py
    (all-background label)

Inputs (adjust paths if they don't match your layout):
  /workspace/pseudolabels/brats_manifest.pkl               -- 200 real cases
  /workspace/pseudolabels/buckets/keep_thr0.95.txt         -- KEEP case_ids
  /workspace/pseudolabels/buckets/bucket_manifest.csv      -- per-case
      inference_mode / n_modalities, to apply the mode-specific policy.
      Auto-detects likely column names; prints what it found if it can't.
  /workspace/pseudolabels/buckets/empty_policy_thr0.95.csv -- output of
      empty_bucket_policy.py (optional -- script proceeds without it if
      missing, just skips the true-negative rows, and says so).
  /workspace/pseudolabels/prob_maps/                       -- per-case
      continuous prob maps, matched to case_id by substring.

Output:
  /workspace/pseudolabels/training_cohort.csv
    columns: case_id, source, image_paths_ref, label_type, label_path,
             threshold, weight
    label_type is one of: 'gt_seg' (real BraTS, use as-is) or
    'prob_map_thr' (Yale, binarize label_path at `threshold`) or
    'empty' (Yale true-negative, all-background label).
"""
import sys
import glob
import pickle
from pathlib import Path

import pandas as pd

BRATS_MANIFEST_PKL = Path('/workspace/pseudolabels/brats_manifest.pkl')
KEEP_LIST_TXT = Path('/workspace/pseudolabels/buckets/keep_thr0.95.txt')
BUCKET_MANIFEST_CSV = Path('/workspace/pseudolabels/buckets/bucket_manifest.csv')
EMPTY_POLICY_CSV = Path('/workspace/pseudolabels/buckets/empty_policy_thr0.95.csv')
PROB_MAPS_DIR = Path('/workspace/pseudolabels/prob_maps')
OUT_CSV = Path('/workspace/pseudolabels/training_cohort.csv')

# THRESHOLD DECISION LOG:
# Switched from 0.85 to 0.95, per threshold_recommendation.txt's own
# criterion (RECOMMENDED THRESHOLD by full-input Dice: 0.95). Full-input
# Dice vs BraTS GT was statistically flat across 0.85/0.90/0.95
# (mean 0.481/0.482/0.483, median ~0.55/~0.55/~0.56) -- raising the
# threshold bought no accuracy, but 0.95 gives the tightest confidence
# bucket with no downside on measured Dice. An earlier version of this
# script used 0.85 for its higher KEEP recall (61.1% vs 57.3% at 0.95,
# ~51 extra cases) on the reasoning that more data at equal quality is
# preferable; this was overridden in favor of following the validation
# script's own recommendation directly rather than a manual judgment call,
# since the recommendation is what a reviewer will expect this pipeline
# to follow. If reverting to 0.85, also revert KEEP_LIST_TXT and
# EMPTY_POLICY_CSV above, and re-run empty_bucket_policy.py at THRESHOLD 0.85
# to regenerate empty_policy_thr0.85.csv (LOW_CONF_THRESH was recalibrated
# per-threshold -- 0.70 for 0.85, 0.87 for 0.95 -- these do NOT interchange).
THRESHOLD = 0.95

# 'exclude' drops t1c-fla-only pseudo-labels entirely (recommended default,
# matches the consistently-worse Dice finding). 'downweight' keeps them at
# FLA_ONLY_WEIGHT instead.
FLA_ONLY_POLICY = 'exclude'
FLA_ONLY_WEIGHT = 0.5


def load_brats_rows() -> pd.DataFrame:
    if not BRATS_MANIFEST_PKL.exists():
        sys.exit(f"ERROR: {BRATS_MANIFEST_PKL} not found.")
    manifest = pickle.load(open(BRATS_MANIFEST_PKL, 'rb'))
    rows = []
    for case in manifest:
        rows.append({
            'case_id': case['case_id'],
            'source': 'real_brats',
            'image_paths_ref': case.get('t1c', ''),
            'label_type': 'gt_seg',
            'label_path': case.get('seg', ''),
            'threshold': '',
            'weight': 1.0,
        })
    print(f"Loaded {len(rows)} real BraTS cases")
    return pd.DataFrame(rows)


def load_keep_list() -> set:
    if not KEEP_LIST_TXT.exists():
        sys.exit(f"ERROR: {KEEP_LIST_TXT} not found.")
    ids = set(x.strip() for x in open(KEEP_LIST_TXT) if x.strip())
    print(f"Loaded {len(ids)} KEEP-bucket case_ids at thr={THRESHOLD}")
    return ids


def load_bucket_manifest_modes() -> dict:
    """Returns {case_id: mode_str} where mode_str is one of
    'full' / 'partial_t1c_t1n' / 'partial_t1c_fla' / 'unknown'."""
    if not BUCKET_MANIFEST_CSV.exists():
        print(f"WARNING: {BUCKET_MANIFEST_CSV} not found -- cannot apply "
              f"mode-specific partial policy, all KEEP cases will be treated as 'full'.")
        return {}

    df = pd.read_csv(BUCKET_MANIFEST_CSV)
    print(f"Loaded bucket manifest for mode lookup: columns = {list(df.columns)}")

    case_col = next((c for c in df.columns if c.lower() in ('case_id', 'caseid', 'case')), None)
    mode_col = next((c for c in df.columns if 'inference_mode' in c.lower() or c.lower() == 'mode'), None)
    nmod_col = next((c for c in df.columns if 'n_modalit' in c.lower() or 'n_sequence' in c.lower()), None)

    mapping = {}
    for _, row in df.iterrows():
        cid = row[case_col]
        if mode_col is not None:
            mode_val = str(row[mode_col]).lower()
            if 't1c' in mode_val and 'fla' in mode_val and 't1n' not in mode_val and 't2' not in mode_val:
                mapping[cid] = 'partial_t1c_fla'
            elif ('t1n' in mode_val or 't1-t1c' in mode_val) and 'fla' not in mode_val and 't2' not in mode_val:
                mapping[cid] = 'partial_t1c_t1n'
            elif nmod_col is not None and row.get(nmod_col, 4) >= 4:
                mapping[cid] = 'full'
            else:
                mapping[cid] = 'full' if 'fla' in mode_val and 't2' in mode_val else 'unknown'
        elif nmod_col is not None:
            mapping[cid] = 'full' if row[nmod_col] >= 4 else 'unknown'
        else:
            mapping[cid] = 'unknown'

    n_full = sum(1 for v in mapping.values() if v == 'full')
    n_t1n = sum(1 for v in mapping.values() if v == 'partial_t1c_t1n')
    n_fla = sum(1 for v in mapping.values() if v == 'partial_t1c_fla')
    n_unk = sum(1 for v in mapping.values() if v == 'unknown')
    print(f"Mode breakdown: full={n_full}, partial_t1c_t1n={n_t1n}, "
          f"partial_t1c_fla={n_fla}, unknown={n_unk}")
    return mapping


def find_prob_map(case_id: str):
    matches = glob.glob(str(PROB_MAPS_DIR / f"*{case_id}*"))
    matches = [m for m in matches if m.endswith('.nii.gz') or m.endswith('.nii')]
    return matches[0] if matches else None


def build_yale_pseudo_rows(keep_ids: set, mode_map: dict) -> pd.DataFrame:
    rows = []
    n_excluded_fla = 0
    n_missing_prob = 0
    for cid in sorted(keep_ids):
        mode = mode_map.get(cid, 'unknown')

        if mode == 'partial_t1c_fla' and FLA_ONLY_POLICY == 'exclude':
            n_excluded_fla += 1
            continue

        weight = 1.0
        if mode == 'partial_t1c_fla' and FLA_ONLY_POLICY == 'downweight':
            weight = FLA_ONLY_WEIGHT

        prob_path = find_prob_map(cid)
        if prob_path is None:
            n_missing_prob += 1
            continue

        rows.append({
            'case_id': cid,
            'source': f'yale_pseudo_{mode}',
            'image_paths_ref': '',
            'label_type': 'prob_map_thr',
            'label_path': prob_path,
            'threshold': THRESHOLD,
            'weight': weight,
        })

    print(f"Yale pseudo-label rows built: {len(rows)}")
    print(f"  Excluded ({FLA_ONLY_POLICY}='exclude', mode=partial_t1c_fla): {n_excluded_fla}")
    if n_missing_prob:
        print(f"  WARNING: {n_missing_prob} KEEP cases had no matching prob map file")
    return pd.DataFrame(rows)


def build_empty_true_negative_rows() -> pd.DataFrame:
    if not EMPTY_POLICY_CSV.exists():
        print(f"NOTE: {EMPTY_POLICY_CSV} not found (run empty_bucket_policy.py first "
              f"if you want true-negative EMPTY cases included) -- skipping.")
        return pd.DataFrame([])

    df = pd.read_csv(EMPTY_POLICY_CSV)
    true_neg = df[df['decision'] == 'TRUE_NEGATIVE']
    rows = []
    for _, row in true_neg.iterrows():
        rows.append({
            'case_id': row['case_id'],
            'source': 'yale_true_negative',
            'image_paths_ref': '',
            'label_type': 'empty',
            'label_path': '',
            'threshold': '',
            'weight': 1.0,
        })
    print(f"True-negative EMPTY rows included: {len(rows)}")
    return pd.DataFrame(rows)


def main():
    brats_df = load_brats_rows()
    keep_ids = load_keep_list()
    mode_map = load_bucket_manifest_modes()
    yale_df = build_yale_pseudo_rows(keep_ids, mode_map)
    empty_df = build_empty_true_negative_rows()

    full_df = pd.concat([brats_df, yale_df, empty_df], ignore_index=True)

    print("\n=== FINAL TRAINING COHORT SUMMARY ===")
    print(full_df['source'].value_counts().to_string())
    print(f"\nTotal training examples: {len(full_df)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == '__main__':
    main()