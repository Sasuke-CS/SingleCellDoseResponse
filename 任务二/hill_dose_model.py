# -*- coding: utf-8 -*-
"""Training-only, drug-specific normalized Hill dose-response scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class HillDoseModel:
    hill_coefficient: float
    ec50_nM: float
    max_dose_nM: float
    fit_rmse: float

    def alpha(self, dose: float) -> float:
        """Hill response normalized so alpha(max_dose) is exactly one."""
        dose = float(dose)
        if dose < 0 or dose > self.max_dose_nM:
            raise ValueError("dose must be between zero and max_dose_nM")
        if dose == 0:
            return 0.0
        h = self.hill_coefficient
        ec50 = self.ec50_nM
        raw = dose**h / (ec50**h + dose**h)
        raw_max = self.max_dose_nM**h / (ec50**h + self.max_dose_nM**h)
        return float(raw / raw_max)


def fit_normalized_hill(
    doses_nM: Iterable[float],
    response_ratios: Iterable[float],
    max_dose_nM: float,
) -> HillDoseModel:
    """Fit h and EC50 with robust loss; no target-cell response is required."""
    doses = np.asarray(list(doses_nM), dtype=float)
    ratios = np.asarray(list(response_ratios), dtype=float)
    valid = np.isfinite(doses) & np.isfinite(ratios) & (doses > 0)
    doses, ratios = doses[valid], ratios[valid]
    if len(doses) < 3:
        raise ValueError("At least three valid training dose observations are required")

    def normalized_hill(params: np.ndarray) -> np.ndarray:
        h, log10_ec50 = params
        ec50 = 10.0**log10_ec50
        raw = doses**h / (ec50**h + doses**h)
        raw_max = max_dose_nM**h / (ec50**h + max_dose_nM**h)
        return raw / raw_max

    fit = least_squares(
        lambda params: normalized_hill(params) - ratios,
        x0=np.array([1.0, np.log10(np.median(doses))]),
        bounds=(np.array([0.1, -3.0]), np.array([5.0, 7.0])),
        loss="soft_l1",
        f_scale=0.15,
        max_nfev=5000,
    )
    predicted = normalized_hill(fit.x)
    rmse = float(np.sqrt(np.mean((predicted - ratios) ** 2)))
    return HillDoseModel(
        hill_coefficient=float(fit.x[0]),
        ec50_nM=float(10.0 ** fit.x[1]),
        max_dose_nM=float(max_dose_nM),
        fit_rmse=rmse,
    )

