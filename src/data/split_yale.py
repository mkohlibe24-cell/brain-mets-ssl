"""
Patient-level train/val/test split of the Yale unlabeled pseudo-label cohort
(1362 cases from cohort.pkl). Groups by patient_id (case_id prefix before the
trailing visit date) to guarantee zero patient overlap across splits, even
though cohort.pkl should already be one-visit-per-patient.
"""

import pickle
import re
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

COHORT_PKL = '/workspace/pseudolabels/cohort.pkl'
OUT_DIR = Path('/workspace/pseudolabels/splits')
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

OUT_DIR.mkdir(parents=True, exist_ok=True)

cohort = pickle.load(open(COHORT_PKL, 'rb'))
print(f"Total Yale cases in cohort.pkl: {len(cohort)}")

case_ids = [c['case_id'] for c in cohort]

# patient_id = everything before the trailing _YYYY-MM-DD date
def get_patient_id(case_id):
    return re.sub(r'_\d{4}-\d{2}-\d{2}$', '', case_id)

patient_ids = [get_patient_id(cid) for cid in case_ids]
n_unique_patients = len(set(patient_ids))
print(f"Unique patients: {n_unique_patients}")

# First split: train vs temp(val+test)
gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=SEED)
train_idx, temp_idx = next(gss1.split(case_ids, groups=patient_ids))

temp_case_ids = [case_ids[i] for i in temp_idx]
temp_patient_ids = [patient_ids[i] for i in temp_idx]

# Second split: val vs test out of temp
gss2 = GroupShuffleSplit(n_splits=1, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), random_state=SEED)
val_idx_rel, test_idx_rel = next(gss2.split(temp_case_ids, groups=temp_patient_ids))

train_ids = [case_ids[i] for i in train_idx]
val_ids   = [temp_case_ids[i] for i in val_idx_rel]
test_ids  = [temp_case_ids[i] for i in test_idx_rel]

print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")

# Sanity: zero patient overlap
train_p = set(get_patient_id(c) for c in train_ids)
val_p   = set(get_patient_id(c) for c in val_ids)
test_p  = set(get_patient_id(c) for c in test_ids)
assert not (train_p & val_p), "Train/Val patient overlap!"
assert not (train_p & test_p), "Train/Test patient overlap!"
assert not (val_p & test_p), "Val/Test patient overlap!"
print("Zero patient overlap confirmed.")

for name, ids in [('yale_train', train_ids), ('yale_val', val_ids), ('yale_test', test_ids)]:
    out_path = OUT_DIR / f'{name}.txt'
    with open(out_path, 'w') as f:
        f.write('\n'.join(sorted(ids)))
    print(f"Wrote {len(ids)} case_ids -> {out_path}")

print("Done.")
