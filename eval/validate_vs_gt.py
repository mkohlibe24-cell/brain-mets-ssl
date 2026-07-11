"""
Validates patched-AURORA continuous pseudo-labels against 200 real
BraTS-MET ground-truth segmentations across full and simulated partial
input modes, at 3 thresholds.
Output: /workspace/eval/validation_results.csv
"""
import sys
import csv
import traceback
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.insert(0, '/workspace/src/pseudolabel')
import aurora_patch

from brainles_aurora.inferer import AuroraInferer, AuroraInfererConfig

MANIFEST_PKL = '/workspace/pseudolabels/brats_manifest.pkl'
OUT_CSV = Path('/workspace/eval/validation_results.csv')
THRESHOLDS = [0.85, 0.90, 0.95]

MODES = {
    'full': {'t1': 't1n', 't1c': 't1c', 't2': 't2w', 'fla': 't2f'},
    'partial_t1c_t1n': {'t1': 't1n', 't1c': 't1c', 't2': None, 'fla': None},
    'partial_t1c_fla': {'t1': None, 't1c': 't1c', 't2': None, 'fla': 't2f'},
}


def dice_score(pred_binary, gt_binary):
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def load_done_keys():
    if not OUT_CSV.exists():
        return set()
    done = set()
    with open(OUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row['case_id'], row['mode'], row['threshold']))
    return done


def append_rows(rows):
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'mode', 'threshold', 'dice', 'n_modalities_used'])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = pickle.load(open(MANIFEST_PKL, 'rb'))
    print(f"Loaded {len(manifest)} BraTS cases")

    done_keys = load_done_keys()
    print(f"Already completed rows: {len(done_keys)}")

    cfg = AuroraInfererConfig(tta=False, cuda_devices='0', threshold=0.5)
    inferer = AuroraInferer(config=cfg)

    n_ok, n_fail = 0, 0

    for i, case in enumerate(manifest):
        cid = case['case_id']
        gt_path = case['seg']
        gt_img = None

        for mode_name, key_map in MODES.items():
            if all((cid, mode_name, str(t)) in done_keys for t in THRESHOLDS):
                continue

            try:
                infer_kwargs = {}
                n_mods = 0
                for aurora_key, brats_mod in key_map.items():
                    if brats_mod is not None and case.get(brats_mod):
                        infer_kwargs[aurora_key] = case[brats_mod]
                        n_mods += 1
                    else:
                        infer_kwargs[aurora_key] = None

                if infer_kwargs.get('t1c') is None:
                    print(f"[{cid}] {mode_name}: no t1c, skipping")
                    continue

                out = inferer.infer(**infer_kwargs)
                prob = np.asarray(out['metastasis_network_floats'])

                if gt_img is None:
                    gt_img = nib.load(gt_path).get_fdata()
                    gt_binary_any = (gt_img > 0).astype(np.uint8)

                rows = []
                for thr in THRESHOLDS:
                    pred_binary = (prob >= thr).astype(np.uint8)
                    d = dice_score(pred_binary, gt_binary_any)
                    rows.append({
                        'case_id': cid, 'mode': mode_name, 'threshold': str(thr),
                        'dice': f"{d:.4f}", 'n_modalities_used': n_mods
                    })
                append_rows(rows)
                n_ok += 1

            except Exception as e:
                print(f"[{cid}] {mode_name} FAILED: {e}")
                traceback.print_exc()
                n_fail += 1

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(manifest)} cases (ok={n_ok}, fail={n_fail})")

    print(f"\nDone. ok={n_ok} fail={n_fail}")
    print(f"Results -> {OUT_CSV}")


if __name__ == '__main__':
    main()
