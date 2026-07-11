"""
Monkey-patches brainles_aurora's ModelHandler._post_process to also return
the pre-threshold sigmoid activation (continuous confidence, 0-1) alongside
the binarized output. The pretrained AURORA release binarizes its metastasis
output internally; this patch preserves the raw floats for downstream
thresholding/bucketing at multiple confidence levels.

Usage:
    import aurora_patch  # applies the patch on import
"""

import numpy as np
from brainles_aurora.inferer.model import ModelHandler
from brainles_aurora.inferer.constants import Output


def _post_process_with_floats(self, onehot_model_outputs_CHWD):
    activated = onehot_model_outputs_CHWD[0].sigmoid().detach().cpu().numpy()  # (C,H,W,D) floats
    binarized = (activated >= self.config.threshold).astype(np.uint8)

    whole, enh = binarized[0], binarized[1]
    final_seg = whole.copy()
    final_seg[whole == 1] = 1
    final_seg[enh == 1] = 2

    return {
        Output.SEGMENTATION: final_seg,
        Output.WHOLE_NETWORK: binarized[0],
        Output.METASTASIS_NETWORK: binarized[1],
        # continuous sigmoid probabilities (what we actually want for bucketing/validation)
        "whole_network_floats": activated[0],
        "metastasis_network_floats": activated[1],
    }


# apply the patch globally
ModelHandler._post_process = _post_process_with_floats
print('patched ModelHandler._post_process')
