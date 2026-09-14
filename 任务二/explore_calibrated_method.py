#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Explore a training-only gene-space calibrated monotone dose scaler."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_task2 import (  # noqa: E402
    CONTROL, DOSES, DRUGS, MAX_DOSE, TARGET, build_results, condition_name,
    decode, load_preserved_model, mean_rows, seed_everything, target_maximum_delta,
)
from original_scvidr_loglinear import alpha_loglinear
import scanpy as sc  # noqa: E402


def fit_empirical_alphas(model, train, latent):
    grid = np.linspace(0.0, 1.20, 61)
    raw_rows = []
    fitted = {}
    obs = train.obs
    for drug in DRUGS:
        raw_alpha = []
        for dose in DOSES[:-1]:
            losses = []
            for alpha in grid:
                line_losses = []
                for line in ["K562", "MCF7"]:
                    ctrl_mask = (obs.cell_line.to_numpy() == line) & (obs.perturbation.to_numpy() == CONTROL)
                    high_mask = (obs.cell_line.to_numpy() == line) & (obs.perturbation.to_numpy() == drug) & (obs.dose_value.to_numpy() == MAX_DOSE)
                    dose_mask = (obs.cell_line.to_numpy() == line) & (obs.perturbation.to_numpy() == drug) & (obs.dose_value.to_numpy() == dose)
                    ctrl_latent = latent[ctrl_mask]
                    delta = mean_rows(latent[high_mask]) - mean_rows(ctrl_latent)
                    pred_mean = decode(model, ctrl_latent + alpha * delta).mean(axis=0)
                    true_mean = np.asarray(train.X[dose_mask].mean(axis=0)).reshape(-1)
                    line_losses.append(float(np.mean((pred_mean - true_mean) ** 2)))
                losses.append(np.mean(line_losses))
            best = int(np.argmin(losses))
            raw_alpha.append(float(grid[best]))
            raw_rows.append({"drug": drug, "dose_nM": dose, "raw_best_alpha": grid[best], "training_mse": losses[best]})
        x = np.log10(np.asarray(DOSES, dtype=float))
        y = np.asarray(raw_alpha + [1.0])
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        monotone = iso.fit_transform(x, y)
        monotone[-1] = 1.0
        fitted[drug] = {dose: float(alpha) for dose, alpha in zip(DOSES, monotone)}
        for row in raw_rows:
            if row["drug"] == drug:
                row["monotone_alpha"] = fitted[drug][row["dose_nM"]]
        raw_rows.append({"drug": drug, "dose_nM": MAX_DOSE, "raw_best_alpha": 1.0, "training_mse": np.nan, "monotone_alpha": 1.0})
    return fitted, pd.DataFrame(raw_rows)


def predict_calibrated(model, train, latent, fitted):
    import anndata as ad
    pieces = []
    for drug in DRUGS:
        ctrl_latent, delta = target_maximum_delta(latent, train.obs, drug)
        for dose in DOSES:
            alpha = fitted[drug][dose]
            x = decode(model, ctrl_latent + alpha * delta)
            obs = pd.DataFrame({"cell_line": TARGET, "perturbation": drug, "dose_value": dose,
                "dose_unit": "nM", "time": 24.0, "condition": condition_name(drug, dose),
                "source": "calibrated_isotonic", "alpha": alpha},
                index=[f"calibrated_{drug}_{dose:g}_{i}" for i in range(len(x))])
            pieces.append(ad.AnnData(X=x, obs=obs, var=train.var.copy()))
    return ad.concat(pieces, join="inner", merge="same", index_unique=None)


def evaluate(dataset, pred):
    rows = []
    ctrl = dataset[(dataset.obs.cell_line == TARGET) & (dataset.obs.perturbation == CONTROL)]
    ctrl_mean = np.asarray(ctrl.X.mean(axis=0)).reshape(-1)
    hvg = dataset.var.hvg_rank.to_numpy() <= 500
    for drug in DRUGS:
        for dose in DOSES:
            truth = dataset[(dataset.obs.cell_line == TARGET) & (dataset.obs.perturbation == drug) & (dataset.obs.dose_value == dose)]
            p = pred[(pred.obs.perturbation == drug) & (pred.obs.dose_value == dose)]
            tm = np.asarray(truth.X.mean(axis=0)).reshape(-1); pm = np.asarray(p.X.mean(axis=0)).reshape(-1)
            de = np.zeros(dataset.n_vars, bool); de[np.argsort(-np.abs(tm-ctrl_mean))[:100]] = True
            for name, mask in {"all_retained_genes": np.ones(dataset.n_vars,bool), "top500_HVG":hvg, "top100_DE":de}.items():
                rows.append({"drug":drug,"dose_nM":dose,"gene_set":name,
                    "pearson_calibrated":stats.pearsonr(tm[mask],pm[mask])[0],
                    "r2_calibrated":r2_score(tm[mask],pm[mask])})
    return pd.DataFrame(rows)


def fit_training_residuals(model, train, latent):
    """Average gene-level log-linear residual shared by the two training lines."""
    obs = train.obs
    residuals = {}
    diagnostics = []
    for drug in DRUGS:
        for dose in DOSES:
            line_residuals = []
            for line in ["K562", "MCF7"]:
                ctrl_mask = (obs.cell_line.to_numpy() == line) & (obs.perturbation.to_numpy() == CONTROL)
                high_mask = (obs.cell_line.to_numpy() == line) & (obs.perturbation.to_numpy() == drug) & (obs.dose_value.to_numpy() == MAX_DOSE)
                dose_mask = (obs.cell_line.to_numpy() == line) & (obs.perturbation.to_numpy() == drug) & (obs.dose_value.to_numpy() == dose)
                ctrl_latent = latent[ctrl_mask]
                delta = mean_rows(latent[high_mask]) - mean_rows(ctrl_latent)
                pred_mean = decode(model, ctrl_latent + alpha_loglinear(dose, MAX_DOSE) * delta).mean(axis=0)
                true_mean = np.asarray(train.X[dose_mask].mean(axis=0)).reshape(-1)
                line_residuals.append(true_mean - pred_mean)
            r1, r2 = line_residuals
            # Agreement-weighted shrinkage: transferable residuals receive more weight.
            agreement = np.exp(-np.abs(r1 - r2) / 0.25)
            correction = 0.5 * (r1 + r2) * agreement
            residuals[(drug, dose)] = correction
            diagnostics.append({"drug":drug,"dose_nM":dose,"mean_abs_correction":np.mean(np.abs(correction)),
                "training_residual_correlation":stats.pearsonr(r1,r2)[0]})
    return residuals, pd.DataFrame(diagnostics)


def predict_residual_transfer(model, train, latent, residuals):
    import anndata as ad
    pieces=[]
    for drug in DRUGS:
        ctrl_latent, delta = target_maximum_delta(latent, train.obs, drug)
        for dose in DOSES:
            alpha=alpha_loglinear(dose,MAX_DOSE)
            x=decode(model,ctrl_latent+alpha*delta)+residuals[(drug,dose)][None,:]
            obs=pd.DataFrame({"cell_line":TARGET,"perturbation":drug,"dose_value":dose,"source":"residual_transfer"},
                index=[f"residual_{drug}_{dose:g}_{i}" for i in range(len(x))])
            pieces.append(ad.AnnData(X=x,obs=obs,var=train.var.copy()))
    return ad.concat(pieces,join="inner",merge="same",index_unique=None)


def main():
    seed_everything()
    dataset = sc.read_h5ad(HERE.parent / "任务一" / "task1_preprocessed.h5ad")
    model, train = load_preserved_model(dataset, HERE.parent / "任务一" / "model")
    latent = model.get_latent_representation(train)
    fitted, fit_table = fit_empirical_alphas(model, train, latent)
    pred = predict_calibrated(model, train, latent, fitted)
    metrics = evaluate(dataset, pred)
    baseline = pd.read_csv(HERE / "metrics_comparison.csv")
    merged = baseline.merge(metrics, on=["drug","dose_nM","gene_set"])
    merged["r2_gain_vs_loglinear"] = merged.r2_calibrated - merged.r2_original_loglinear
    merged["pearson_gain_vs_loglinear"] = merged.pearson_calibrated - merged.pearson_original_loglinear
    fit_table.to_csv(HERE / "calibrated_alpha_exploration.csv", index=False)
    merged.to_csv(HERE / "calibrated_method_exploration_metrics.csv", index=False)
    print(json.dumps(fitted, indent=2))
    print(merged.groupby('gene_set')[["r2_gain_vs_loglinear","pearson_gain_vs_loglinear"]].agg(['mean',lambda x:(x>0).sum()]))
    residuals, residual_diagnostics = fit_training_residuals(model, train, latent)
    residual_pred = predict_residual_transfer(model, train, latent, residuals)
    residual_metrics = evaluate(dataset, residual_pred).rename(columns={"pearson_calibrated":"pearson_residual","r2_calibrated":"r2_residual"})
    residual_compare = baseline.merge(residual_metrics,on=["drug","dose_nM","gene_set"])
    residual_compare["r2_gain_vs_loglinear"] = residual_compare.r2_residual-residual_compare.r2_original_loglinear
    residual_compare["pearson_gain_vs_loglinear"] = residual_compare.pearson_residual-residual_compare.pearson_original_loglinear
    residual_diagnostics.to_csv(HERE/"residual_transfer_diagnostics.csv",index=False)
    residual_compare.to_csv(HERE/"residual_transfer_exploration_metrics.csv",index=False)
    print("RESIDUAL TRANSFER")
    print(residual_compare.groupby('gene_set')[["r2_gain_vs_loglinear","pearson_gain_vs_loglinear"]].agg(['mean',lambda x:(x>0).sum()]))


if __name__ == "__main__":
    main()
