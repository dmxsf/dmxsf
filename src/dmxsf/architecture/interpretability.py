"""Interpretability: superposed features and causal patching.

Wiki pages
  Chinese: https://dmxsf.com/wiki/frontier/interpretability/
  English: https://dmxsf.com/en/wiki/frontier/interpretability/
"""

import numpy as np


def feature_scores(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Encode a sum of feature directions and decode with their dot products."""
    return features @ (coefficients @ features)


def patch_feature(clean: np.ndarray, corrupted: np.ndarray, feature_index: int) -> np.ndarray:
    """Replace one named activation in an instrumented toy model."""
    patched = corrupted.copy()
    patched[feature_index] = clean[feature_index]
    return patched
