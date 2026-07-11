"""
Patient-level train/val/test split of the 200 BraTS-MET cases.
Scans the Pretreat-MetsToBrain-Masks folder directly.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split

BRATS_ROOT = Path('/workspace/PKG - Pretreat-MetsToBrain-Masks/Pretreat-MetsToBrain-Masks')
OUT_DIR = Path('/workspace/pseudolabels/splits')
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

OUT_DIR.mkdir(parents=True, exist_ok=True)

brats_cases = sorted([p.name for p in BRATS_ROOT.iterdir() if p.is_dir() and p.name.startswith('BraTS-MET')])

print(f"Total BraTS cases found: {len(brats_cases)}")
assert len(brats_cases) == 200, f"Expected 200 BraTS cases, got {len(brats_cases)}"

train_ids, temp_ids = train_test_split(
    brats_cases, train_size=TRAIN_FRAC, random_state=SEED, shuffle=True
)
val_ids, test_ids = train_test_split(
    temp_ids, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), random_state=SEED, shuffle=True
)

print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
assert len(set(train_ids) & set(val_ids) & set(test_ids)) == 0, "Overlap detected!"

for name, ids in [('brats_train', train_ids), ('brats_val', val_ids), ('brats_test', test_ids)]:
    out_path = OUT_DIR / f'{name}.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(sorted(ids)))
    print(f"Wrote {len(ids)} case_ids -> {out_path}")

print("Done.")
