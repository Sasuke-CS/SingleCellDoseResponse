# -*- coding: utf-8 -*-
"""Dose-specific latent direction regression (DSDR).

Unlike scalar interpolation from a single maximum-dose direction, DSDR learns
the full perturbation vector independently at every observed training dose.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression


def predict_target_delta(
    latent: np.ndarray,
    cell_lines: np.ndarray,
    perturbations: np.ndarray,
    doses: np.ndarray,
    drug: str,
    dose: float,
    target_cell_line: str,
    source_cell_lines=("K562", "MCF7"),
    control_label="control",
):
    """Regress source dose-specific deltas on control centroids and extrapolate."""
    centroids, deltas = [], []
    for line in source_cell_lines:
        ctrl = (cell_lines == line) & (perturbations == control_label)
        treated = (cell_lines == line) & (perturbations == drug) & (doses == dose)
        ctrl_centroid = latent[ctrl].mean(axis=0)
        centroids.append(ctrl_centroid)
        deltas.append(latent[treated].mean(axis=0) - ctrl_centroid)
    target_ctrl = (cell_lines == target_cell_line) & (perturbations == control_label)
    target_control_latent = latent[target_ctrl]
    regression = LinearRegression().fit(np.asarray(centroids), np.asarray(deltas))
    target_delta = regression.predict([target_control_latent.mean(axis=0)])[0]
    return target_control_latent, target_delta

