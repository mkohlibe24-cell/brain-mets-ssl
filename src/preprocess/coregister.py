"""
Rigid-registers t1/t2/flair onto t1c using ANTs. Used for Yale's raw
(non-co-registered) clinical scans before AURORA inference. BraTS does
not need this step — it ships pre-registered.
"""

import ants
from pathlib import Path

COREG_DIR = Path('/workspace/pseudolabels/coreg_tmp')  # adjust as needed per caller


def coregister(case, coreg_dir=COREG_DIR):
    """Rigid-register t1/t2/flair onto t1c. Returns dict of aligned paths."""
    cid = case['case_id']
    cdir = coreg_dir / cid
    cdir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(case['t1c'])
    aligned = {'t1c': case['t1c']}

    for mod in ('t1', 't2', 'flair'):
        src = case.get(mod)
        if not src:
            aligned[mod] = None
            continue
        moving = ants.image_read(src)
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid')
        out_path = cdir / f'{mod}_aligned.nii.gz'
        ants.image_write(reg['warpedmovout'], str(out_path))
        aligned[mod] = str(out_path)

    return aligned
