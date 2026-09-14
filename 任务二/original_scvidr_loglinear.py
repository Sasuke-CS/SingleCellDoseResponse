# -*- coding: utf-8 -*-
"""Original scVIDR log-linear dose interpolation, kept as a standalone baseline.

This is the equation used by ``VIDR.predict(..., continuous=True)`` in the
upstream repository. Keeping it separate makes the baseline explicit and keeps
the improved dose model from silently changing the original implementation.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def alpha_loglinear(dose: float, max_dose: float) -> float:
    """Return alpha_log(d) = log(1+d) / log(1+d_max)."""
    if dose < 0 or max_dose <= 0 or dose > max_dose:
        raise ValueError("Require 0 <= dose <= max_dose and max_dose > 0")
    return float(np.log1p(dose) / np.log1p(max_dose))


def predict_latent_loglinear(
    control_latent: np.ndarray,
    maximum_dose_delta: np.ndarray,
    doses: Iterable[float],
) -> Mapping[float, np.ndarray]:
    """Apply the original scVIDR interpolation in latent space."""
    doses = [float(d) for d in doses]
    max_dose = max(doses)
    return {
        dose: control_latent
        + alpha_loglinear(dose, max_dose) * maximum_dose_delta
        for dose in doses
    }

