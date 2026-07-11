"""
Builds a manifest of all 200 BraTS-MET cases with verified paths to each
modality file (t1c/t1n/t2f/t2w/seg). Checks all 5 files exist per case
before AURORA inference. No ANTs coreg needed — BraTS is pre-registered.
"""

import pickle
from pathlib import Path

BRATS_ROOT = Path('/workspace/PKG - Pretreat-MetsToBrain-Masks/Pretreat-MetsToBrain-Masks')
OUT_PKL = Path('/workspace/pseudolabels/brats_manifest.pkl')

MODALITIES = ['t1c', 't1n', 't2f', 't2w', 'seg']

manifest = []
missing_cases = []

case_dirs = sorted([p for p in BRATS_ROOT.iterdir() if p.is_dir() and p.name.startswith('BraTS-MET')])
print(f"Found {len(case_dirs)} case folders")

for case_dir in case_dirs:
    case_id = case_dir.name
    entry = {'case_id': case_id}
    all_found = True

    for mod in MODALITIES:
        expected_file = case_dir / f"{case_id}-{mod}.nii.gz"
        if expected_file.exists():
            entry[mod] = str(expected_file)
        else:
            entry[mod] = None
            all_found = False

    if all_found:
        manifest.append(entry)
    else:
        missing_cases.append(case_id)

print(f"Complete cases (all 5 files found): {len(manifest)}")
print(f"Incomplete cases: {len(missing_cases)}")
if missing_cases:
    print("Missing:", missing_cases)

with open(OUT_PKL, 'wb') as f:
    pickle.dump(manifest, f)
print(f"Saved manifest -> {OUT_PKL}")
