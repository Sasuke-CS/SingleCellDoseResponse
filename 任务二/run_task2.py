#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 2: compare original scVIDR log-linear interpolation with a Hill model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TASK1 = ROOT / "任务一"
RUNTIME = HERE / ".runtime_cache"
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNTIME))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(RUNTIME))

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(ROOT / "vidr"))
import vidr as vidr_module  # noqa: E402
from scvi.data import setup_anndata  # noqa: E402
from vidr import VIDR  # noqa: E402

sys.path.insert(0, str(HERE))
from hill_dose_model import fit_normalized_hill  # noqa: E402
from original_scvidr_loglinear import alpha_loglinear, predict_latent_loglinear  # noqa: E402

vidr_module.LinearRegression = LinearRegression

SEED = 20260913
TARGET = "A549"
DRUGS = ["(+)-JQ1", "Trametinib (GSK1120212)"]
DOSES = [10.0, 100.0, 1000.0, 10000.0]
MAX_DOSE = max(DOSES)
CONTROL = "control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed", type=Path, default=TASK1 / "task1_preprocessed.h5ad")
    parser.add_argument("--model", type=Path, default=TASK1 / "model")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def condition_name(perturbation: str, dose: float) -> str:
    return "control_0" if perturbation == CONTROL else f"{perturbation}_{dose:g}nM"


def load_preserved_model(dataset: ad.AnnData, model_path: Path) -> tuple[VIDR, ad.AnnData]:
    train = dataset[dataset.obs["split"] == "train"].copy()
    train.obs["condition"] = train.obs["condition"].astype(str).astype("category")
    train.obs["cell_line"] = train.obs["cell_line"].astype(str).astype("category")
    train = setup_anndata(train, copy=True, batch_key="condition", labels_key="cell_line")
    model = VIDR.load(model_path, train, use_gpu=torch.cuda.is_available())
    return model, train


def mean_rows(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).mean(axis=0).reshape(-1)


def fit_dose_models(
    latent: np.ndarray, obs: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one Hill model per drug from non-target cell lines only."""
    models = {}
    ratio_rows = []
    parameter_rows = []
    curve_rows = []
    for drug in DRUGS:
        for cell_line in ["K562", "MCF7"]:
            ctrl_mask = (obs["cell_line"].to_numpy() == cell_line) & (
                obs["perturbation"].to_numpy() == CONTROL
            )
            ctrl_centroid = mean_rows(latent[ctrl_mask])
            deltas = {}
            for dose in DOSES:
                treat_mask = (
                    (obs["cell_line"].to_numpy() == cell_line)
                    & (obs["perturbation"].to_numpy() == drug)
                    & (obs["dose_value"].to_numpy() == dose)
                )
                deltas[dose] = mean_rows(latent[treat_mask]) - ctrl_centroid
            reference = deltas[MAX_DOSE]
            denom = float(np.dot(reference, reference))
            if denom <= 1e-12:
                raise RuntimeError(f"Near-zero maximum-dose delta for {drug}/{cell_line}")
            for dose in DOSES:
                ratio = float(np.dot(deltas[dose], reference) / denom)
                ratio_rows.append(
                    {
                        "drug": drug,
                        "training_cell_line": cell_line,
                        "dose_nM": dose,
                        "projected_response_ratio": ratio,
                    }
                )
        drug_ratios = [r for r in ratio_rows if r["drug"] == drug]
        hill = fit_normalized_hill(
            [r["dose_nM"] for r in drug_ratios],
            [r["projected_response_ratio"] for r in drug_ratios],
            MAX_DOSE,
        )
        models[drug] = hill
        parameter_rows.append(
            {
                "drug": drug,
                "hill_coefficient": hill.hill_coefficient,
                "ec50_nM": hill.ec50_nM,
                "training_fit_rmse": hill.fit_rmse,
                "n_training_points": len(drug_ratios),
            }
        )
        dense_doses = np.geomspace(min(DOSES), MAX_DOSE, 160)
        for dose in dense_doses:
            curve_rows.extend(
                [
                    {"drug": drug, "dose_nM": dose, "method": "original_loglinear", "alpha": alpha_loglinear(dose, MAX_DOSE)},
                    {"drug": drug, "dose_nM": dose, "method": "improved_hill", "alpha": hill.alpha(dose)},
                ]
            )
        # Add exact evaluated points for convenient downstream reuse.
        for ratio in drug_ratios:
            ratio["hill_fitted_alpha"] = hill.alpha(ratio["dose_nM"])
            ratio["loglinear_alpha"] = alpha_loglinear(ratio["dose_nM"], MAX_DOSE)
    return models, pd.DataFrame(ratio_rows), pd.DataFrame(parameter_rows), pd.DataFrame(curve_rows)


def target_maximum_delta(latent: np.ndarray, obs: pd.DataFrame, drug: str) -> tuple[np.ndarray, np.ndarray]:
    """The same centroid-to-delta regression used by scVIDR, at maximum dose."""
    centroids, deltas = [], []
    for cell_line in ["K562", "MCF7"]:
        ctrl_mask = (obs["cell_line"].to_numpy() == cell_line) & (
            obs["perturbation"].to_numpy() == CONTROL
        )
        high_mask = (
            (obs["cell_line"].to_numpy() == cell_line)
            & (obs["perturbation"].to_numpy() == drug)
            & (obs["dose_value"].to_numpy() == MAX_DOSE)
        )
        ctrl_centroid = mean_rows(latent[ctrl_mask])
        centroids.append(ctrl_centroid)
        deltas.append(mean_rows(latent[high_mask]) - ctrl_centroid)
    target_mask = (obs["cell_line"].to_numpy() == TARGET) & (
        obs["perturbation"].to_numpy() == CONTROL
    )
    target_control_latent = latent[target_mask]
    regression = LinearRegression().fit(np.asarray(centroids), np.asarray(deltas))
    predicted_delta = regression.predict([mean_rows(target_control_latent)])[0]
    return target_control_latent, predicted_delta


def decode(model: VIDR, latent: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(latent, dtype=torch.float32)
        return model.module.generative(tensor)["px"].detach().cpu().numpy()


def predict_both_methods(model: VIDR, train: ad.AnnData, latent: np.ndarray, hill_models: dict) -> ad.AnnData:
    pieces = []
    for drug in DRUGS:
        ctrl_latent, delta_max = target_maximum_delta(latent, train.obs, drug)
        original_latents = predict_latent_loglinear(ctrl_latent, delta_max, DOSES)
        improved_latents = {
            dose: ctrl_latent + hill_models[drug].alpha(dose) * delta_max for dose in DOSES
        }
        for method, predictions in [
            ("original_loglinear", original_latents),
            ("improved_hill", improved_latents),
        ]:
            for dose, latent_pred in predictions.items():
                x = decode(model, latent_pred)
                obs = pd.DataFrame(
                    {
                        "cell_line": TARGET,
                        "perturbation": drug,
                        "dose_value": dose,
                        "dose_unit": "nM",
                        "time": 24.0,
                        "condition": condition_name(drug, dose),
                        "source": method,
                        "alpha": alpha_loglinear(dose, MAX_DOSE)
                        if method == "original_loglinear"
                        else hill_models[drug].alpha(dose),
                    },
                    index=[f"{method}_{drug}_{dose:g}_{i}" for i in range(len(x))],
                )
                pieces.append(ad.AnnData(X=x, obs=obs, var=train.var.copy()))
    return ad.concat(pieces, join="inner", merge="same", index_unique=None)


def vector_mean(x) -> np.ndarray:
    return np.asarray(x.mean(axis=0)).reshape(-1)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.pearsonr(x, y)[0])


def evaluate(dataset: ad.AnnData, predicted: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctrl = dataset[(dataset.obs["cell_line"] == TARGET) & (dataset.obs["perturbation"] == CONTROL)]
    ctrl_mean = vector_mean(ctrl.X)
    hvg500 = dataset.var["hvg_rank"].to_numpy() <= 500
    metrics, gene_rows = [], []
    genes = dataset.var_names.to_numpy()
    for drug in DRUGS:
        for dose in DOSES:
            truth = dataset[
                (dataset.obs["cell_line"] == TARGET)
                & (dataset.obs["perturbation"] == drug)
                & (dataset.obs["dose_value"] == dose)
            ]
            true_mean = vector_mean(truth.X)
            de100 = np.zeros(dataset.n_vars, dtype=bool)
            de100[np.argsort(-np.abs(true_mean - ctrl_mean))[:100]] = True
            for method in ["original_loglinear", "improved_hill"]:
                pred = predicted[
                    (predicted.obs["source"] == method)
                    & (predicted.obs["perturbation"] == drug)
                    & (predicted.obs["dose_value"] == dose)
                ]
                pred_mean = vector_mean(pred.X)
                for gene_set, mask in {
                    "all_retained_genes": np.ones(dataset.n_vars, dtype=bool),
                    "top500_HVG": hvg500,
                    "top100_DE": de100,
                }.items():
                    metrics.append(
                        {
                            "drug": drug,
                            "dose_nM": dose,
                            "method": method,
                            "gene_set": gene_set,
                            "n_genes": int(mask.sum()),
                            "pearson": pearson(true_mean[mask], pred_mean[mask]),
                            "r2": float(r2_score(true_mean[mask], pred_mean[mask])),
                            "n_true_cells": truth.n_obs,
                            "n_pred_cells": pred.n_obs,
                        }
                    )
                for i, gene in enumerate(genes):
                    gene_rows.append(
                        {
                            "drug": drug,
                            "dose_nM": dose,
                            "method": method,
                            "gene": gene,
                            "true_mean": true_mean[i],
                            "predicted_mean": pred_mean[i],
                            "control_mean": ctrl_mean[i],
                            "is_top100_DE": bool(de100[i]),
                            "hvg_rank": int(dataset.var["hvg_rank"].iat[i]),
                        }
                    )
    metric_df = pd.DataFrame(metrics)
    wide = metric_df.pivot_table(
        index=["drug", "dose_nM", "gene_set"], columns="method", values=["pearson", "r2"]
    )
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["pearson_improvement"] = wide["pearson_improved_hill"] - wide["pearson_original_loglinear"]
    wide["r2_improvement"] = wide["r2_improved_hill"] - wide["r2_original_loglinear"]
    return metric_df, pd.DataFrame(gene_rows), wide


def build_results(dataset: ad.AnnData, predicted: ad.AnnData) -> ad.AnnData:
    ctrl = dataset[(dataset.obs["cell_line"] == TARGET) & (dataset.obs["perturbation"] == CONTROL)].copy()
    ctrl.obs["source"] = "control"
    truth = dataset[dataset.obs["split"] == "test"].copy()
    truth.obs["source"] = "true"
    result = ad.concat([ctrl, truth, predicted], join="inner", merge="same", index_unique="-")
    for col in ["cell_line", "perturbation", "condition", "source"]:
        result.obs[col] = result.obs[col].astype(str).astype("category")
    return result


def plot_alpha(ratios: pd.DataFrame, curves: pd.DataFrame, fig_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    palette = {"original_loglinear": "#377eb8", "improved_hill": "#e41a1c"}
    for ax, drug in zip(axes, DRUGS):
        curve = curves[curves["drug"] == drug]
        for method, part in curve.groupby("method", sort=False):
            ax.plot(part["dose_nM"], part["alpha"], color=palette[method], label=method.replace("_", " "))
        points = ratios[ratios["drug"] == drug]
        for cell_line, part in points.groupby("training_cell_line"):
            ax.scatter(part["dose_nM"], part["projected_response_ratio"], s=42, label=f"training: {cell_line}")
        ax.set_xscale("log")
        ax.set_xlabel("Dose (nM)")
        ax.set_title(drug.replace(" (GSK1120212)", ""))
        ax.axhline(0, color="black", linewidth=.7)
    axes[0].set_ylabel("Normalized latent response / alpha")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Dose scaling fitted without A549 treated cells")
    fig.tight_layout()
    fig.savefig(fig_dir / "dose_scaling_functions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, fig_dir: Path) -> None:
    main = metrics[metrics["gene_set"] == "all_retained_genes"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    palette = {"original_loglinear": "#377eb8", "improved_hill": "#e41a1c"}
    for col, drug in enumerate(DRUGS):
        frame = main[main["drug"] == drug]
        for method, group in frame.groupby("method", sort=False):
            label = method.replace("original_loglinear", "original log-linear").replace("improved_hill", "improved Hill")
            axes[0, col].plot(group["dose_nM"], group["pearson"], marker="o", color=palette[method], label=label)
            axes[1, col].plot(group["dose_nM"], group["r2"], marker="o", color=palette[method], label=label)
        axes[0, col].set_title(drug.replace(" (GSK1120212)", ""))
        axes[0, col].set_ylabel("Pearson")
        axes[1, col].set_ylabel("R²")
        axes[1, col].set_xlabel("Dose (nM, log scale)")
        for row in range(2):
            axes[row, col].set_xscale("log")
            axes[row, col].legend(fontsize=8)
    fig.suptitle("Original vs improved dose-response prediction (all retained genes)")
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_original_vs_hill.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gene_curves(gene_means: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    selected = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"true": "#222222", "original_loglinear": "#377eb8", "improved_hill": "#e41a1c"}
    for ax, drug in zip(axes, DRUGS):
        frame = gene_means[gene_means["drug"] == drug]
        high = frame[(frame["dose_nM"] == MAX_DOSE) & (frame["method"] == "original_loglinear")].copy()
        high["effect"] = (high["true_mean"] - high["control_mean"]).abs()
        candidates = high.nlargest(100, "effect")["gene"].tolist()
        scored = []
        for candidate in candidates:
            candidate_frame = frame[(frame["gene"] == candidate) & (frame["dose_nM"] < MAX_DOSE)]
            original = candidate_frame[candidate_frame["method"] == "original_loglinear"].sort_values("dose_nM")
            improved = candidate_frame[candidate_frame["method"] == "improved_hill"].sort_values("dose_nM")
            original_rmse = float(np.sqrt(np.mean((original["predicted_mean"].to_numpy() - original["true_mean"].to_numpy()) ** 2)))
            improved_rmse = float(np.sqrt(np.mean((improved["predicted_mean"].to_numpy() - improved["true_mean"].to_numpy()) ** 2)))
            scored.append((original_rmse - improved_rmse, candidate, original_rmse, improved_rmse))
        rmse_gain, gene, original_rmse, improved_rmse = max(scored)
        selected.append(
            {
                "drug": drug,
                "gene": gene,
                "selection": "largest intermediate-dose RMSE gain among top100 high-dose DE genes",
                "original_intermediate_rmse": original_rmse,
                "improved_intermediate_rmse": improved_rmse,
                "rmse_improvement": rmse_gain,
            }
        )
        g = frame[frame["gene"] == gene]
        true_curve = g[g["method"] == "original_loglinear"].sort_values("dose_nM")
        ax.plot(true_curve["dose_nM"], true_curve["true_mean"], marker="o", color=colors["true"], label="true")
        for method in ["original_loglinear", "improved_hill"]:
            part = g[g["method"] == method].sort_values("dose_nM")
            ax.plot(part["dose_nM"], part["predicted_mean"], marker="o", color=colors[method], label=method.replace("_", " "))
        ax.axhline(true_curve["control_mean"].iat[0], color="#888888", linestyle="--", label="control mean")
        ax.set_xscale("log")
        ax.set_xlabel("Dose (nM)")
        ax.set_ylabel("Mean log-normalized expression")
        ax.set_title(f"{drug.replace(' (GSK1120212)', '')}: {gene}")
        ax.legend(fontsize=8)
    fig.suptitle("Representative high-dose DE genes")
    fig.tight_layout()
    fig.savefig(fig_dir / "representative_gene_dose_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(selected)


def plot_umap(result: ad.AnnData, fig_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    chosen = []
    for _, frame in result.obs.groupby(["source", "perturbation", "dose_value"], observed=True):
        idx = result.obs_names.get_indexer(frame.index)
        chosen.extend(rng.choice(idx, size=min(110, len(idx)), replace=False).tolist())
    vis = result[np.asarray(chosen)].copy()
    vis.X = vis.X.astype(np.float32)
    sc.tl.pca(vis, n_comps=30, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(vis, n_neighbors=20, n_pcs=30, random_state=SEED)
    sc.tl.umap(vis, random_state=SEED, min_dist=.35)
    coords = pd.DataFrame(vis.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=vis.obs_names)
    coords = pd.concat([coords, vis.obs.reset_index(drop=True).set_index(coords.index)], axis=1)

    colors = {"control": "#999999", "true": "#222222", "original_loglinear": "#377eb8", "improved_hill": "#e41a1c"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for i, drug in enumerate(DRUGS):
        for j, dose in enumerate(DOSES):
            ax = axes[i, j]
            mask = (coords["source"] == "control") | (
                (coords["perturbation"] == drug) & (coords["dose_value"] == dose)
            )
            frame = coords[mask]
            for source in ["control", "true", "original_loglinear", "improved_hill"]:
                s = frame[frame["source"] == source]
                ax.scatter(s["UMAP1"], s["UMAP2"], s=7, alpha=.5, c=colors[source], label=source.replace("_", " "), rasterized=True)
            if i == 0:
                ax.set_title(f"{dose:g} nM")
            if j == 0:
                ax.set_ylabel(drug.replace(" (GSK1120212)", "") + "\nUMAP2")
            if i == 1:
                ax.set_xlabel("UMAP1")
            ax.set_xticks([])
            ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
    fig.suptitle("Original log-linear and improved Hill predictions in one shared A549 UMAP", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .91])
    fig.savefig(fig_dir / "umap_original_vs_hill.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return coords


def md_table(frame: pd.DataFrame) -> str:
    headers = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, comparison: pd.DataFrame, params: pd.DataFrame, genes: pd.DataFrame, runtime_s: float) -> None:
    main = comparison[comparison["gene_set"] == "all_retained_genes"].copy()
    display = main[["drug", "dose_nM", "pearson_original_loglinear", "pearson_improved_hill", "pearson_improvement", "r2_original_loglinear", "r2_improved_hill", "r2_improvement"]].copy()
    for col in display.columns[2:]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    wins_r2 = int((main["r2_improvement"] > 0).sum())
    wins_p = int((main["pearson_improvement"] > 0).sum())
    by_drug = main.groupby("drug")[["pearson_improvement", "r2_improvement"]].mean().reset_index()
    for col in ["pearson_improvement", "r2_improvement"]:
        by_drug[col] = by_drug[col].map(lambda x: f"{x:+.4f}")
    param_display = params.copy()
    for col in ["hill_coefficient", "ec50_nM", "training_fit_rmse"]:
        param_display[col] = param_display[col].map(lambda x: f"{x:.4f}")
    gene_text = "；".join(f"{r.drug}: {r.gene}" for r in genes.itertuples())
    by_set = comparison.groupby("gene_set")[["pearson_improvement", "r2_improvement"]].mean().reset_index()
    by_set["r2_wins_of_8"] = [
        int((comparison.loc[comparison["gene_set"] == gene_set, "r2_improvement"] > 0).sum())
        for gene_set in by_set["gene_set"]
    ]
    for col in ["pearson_improvement", "r2_improvement"]:
        by_set[col] = by_set[col].map(lambda x: f"{x:+.4f}")
    report = f"""# 任务二：剂量响应建模与方法改进

## 结论

本任务保留并复用任务一已经训练好的 scVIDR VAE，在同一个 A549 完全留出测试集上比较：

1. 原始方法：$\\alpha_{{log}}(d)=\\log(1+d)/\\log(1+d_{{max}})$；
2. 改进方法：只使用 K562、MCF7 训练数据拟合的药物特异性归一化 Hill 函数。

全部 2,000 个保留基因的逐条件结果如下：

{md_table(display)}

Hill 方法在 8 个条件中的 {wins_r2} 个提高 R²、{wins_p} 个提高 Pearson。按药物平均增量：

{md_table(by_drug)}

不同评价基因集合呈现不同结论：

{md_table(by_set)}

在全基因和 Top 500 HVG 上，Hill 平均略差；但在 Top 100 DE 基因上，六个非最高剂量条件的 R² 均提高，平均提升约 0.0075。即改进方法更接近扰动相关基因的表达幅度，却轻微牺牲了由稳定背景基因主导的整体拟合。

最高剂量处两种函数都被约束为 $\\alpha(d_{{max}})=1$，因此 10000 nM 的预测与指标应完全相同；这也是实现正确性的内部检查。改进是否有效主要由中低剂量决定。

![指标比较](figures/metrics_original_vs_hill.png)

## 方法设计

对于训练细胞系 $c$、药物 $p$ 和剂量 $d$，先计算潜在扰动向量 $\\delta_{{c,p,d}}=\\bar z_{{c,p,d}}-\\bar z_{{c,0}}$。再以最高剂量向量为方向，计算投影响应比：

$$r_{{c,p,d}}=\\frac{{\\delta_{{c,p,d}}^T\\delta_{{c,p,d_{{max}}}}}}{{\\|\\delta_{{c,p,d_{{max}}}}\\|_2^2}}.$$

使用 K562、MCF7 共 8 个训练点，以 robust soft-L1 损失拟合：

$$\\alpha_{{Hill}}(d)=\\frac{{d^h/(EC_{{50}}^h+d^h)}}{{d_{{max}}^h/(EC_{{50}}^h+d_{{max}}^h)}}.$$

拟合参数：

{md_table(param_display)}

Trametinib 的 $h$ 与 $EC_{{50}}$ 到达预设下界，反映 K562/MCF7 在 10 nM 已接近最高剂量投影响应；它应解释为“近似平坦/早饱和”的数值形状，而不能当作可靠的生理 EC50 估计。

![剂量函数](figures/dose_scaling_functions.png)

目标 A549 的最高剂量扰动向量仍采用原 scVIDR 的跨细胞系回归得到。原始与改进方法仅替换剂量缩放系数，VAE、训练集、目标对照细胞、最高剂量方向及解码器完全相同，因而比较是配对且可归因于剂量函数。

## 有效与无效情形分析

`metrics_comparison.csv` 给出全部基因、Top 500 HVG、Top 100 DE 三个集合的逐条件差值。具体而言，全基因 R² 仅 JQ1 10 nM 改善；JQ1 100/1000 nM 与 Trametinib 10/100/1000 nM 变差，10000 nM 相同。相反，Top 100 DE 的六个中低剂量 R² 全部改善。这说明 Hill 函数能改善扰动幅度，却没有改善全局跨基因排序。若两个训练细胞系的投影比例不一致，或目标 A549 的剂量形状不同，药物级 Hill 曲线仍可能变差。

全基因高相关性容易被稳定表达背景抬高，因此应优先结合 R² 改变量和 Top 100 DE 结果判断。这里没有用 A549 真实扰动数据选择 Hill 参数，避免了剂量模型的测试泄漏。

## 代表基因剂量曲线

每个药物先取 A549 最高剂量 Top 100 DE 基因，再选择在三个中间剂量上 Hill 相对原始方法 RMSE 改善最大的基因（仅用于事后代表性展示，不参与 Hill 拟合）：{gene_text}。

![代表基因](figures/representative_gene_dose_curves.png)

## UMAP

A549 对照、真实扰动、原始预测和 Hill 预测使用相同 log-normalized 基因空间，合并后只拟合一次 PCA、邻接图和 UMAP。该图用于观察分布而不只是均值。

![共享 UMAP](figures/umap_original_vs_hill.png)

## 原模型保留说明

- 任务一的原模型及代码位于 `../任务一/model/` 与 `../任务一/run_task1.py`，任务二全程只读加载，没有覆盖。
- `original_scvidr_loglinear.py` 独立保留原始 log-linear 公式及潜在空间预测实现。
- `preservation_manifest.json` 记录任务一代码与模型文件的 SHA-256，可验证原文件未被任务二修改。
- `hill_dose_model.py` 单独实现改进方法，避免与原始基线混写。

## 输出文件

- `metrics.csv`：两种方法 × 8 条件 × 3 基因集合的原始指标。
- `metrics_comparison.csv`：配对后的指标与 Hill 相对原始方法的增量。
- `training_response_ratios.csv`、`hill_parameters.csv`、`alpha_curves.csv`：剂量模型拟合数据和参数。
- `per_gene_mean_expression.csv.gz`：逐条件、方法、基因均值。
- `prediction_results_task2.h5ad`：对照、真实、原始预测及改进预测细胞。
- `umap_coordinates.csv`、`representative_genes.csv` 及 `figures/`：可视化数据与图片。
- `run_task2.py`：完整复现入口；运行配置见 `run_config.json`。

## 局限性

仅有两个非目标细胞系用于估计响应形状和最高剂量回归，无法充分估计跨细胞系不确定性；Hill 函数强制单调饱和，不能表达双相或非单调响应；代表基因由测试集事后选择，只能作为描述性展示。此外，VAE 解码细胞的方差结构可能与真实计数不同，均值指标较好不代表完整单细胞分布完全匹配。

## 复现

```powershell
conda activate scVIDR
python "任务二/run_task2.py"
```

本次任务二运行耗时约 {runtime_s / 60:.1f} 分钟。无需重新训练任务一模型。
"""
    (HERE / "README.md").write_text(report, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    files = [TASK1 / "run_task1.py", TASK1 / "model" / "model_params.pt", TASK1 / "model" / "attr.pkl", TASK1 / "model" / "var_names.csv"]
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in files}
    (HERE / "preservation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.report_only:
        metrics = pd.read_csv(HERE / "metrics.csv")
        comparison = pd.read_csv(HERE / "metrics_comparison.csv")
        params = pd.read_csv(HERE / "hill_parameters.csv")
        genes = pd.read_csv(HERE / "representative_genes.csv")
        runtime = json.loads((HERE / "run_config.json").read_text(encoding="utf-8"))["runtime_seconds"]
        write_report(metrics, comparison, params, genes, runtime)
        write_manifest()
        print("Task 2 report rebuilt", flush=True)
        return

    started = time.time()
    seed_everything()
    fig_dir = HERE / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("[1/7] Loading task-1 data and preserved model", flush=True)
    dataset = sc.read_h5ad(args.preprocessed)
    model, train = load_preserved_model(dataset, args.model)
    print("[2/7] Encoding training cells once", flush=True)
    latent = model.get_latent_representation(train)
    print("[3/7] Fitting training-only drug-specific Hill functions", flush=True)
    hill_models, ratios, params, curves = fit_dose_models(latent, train.obs)
    ratios.to_csv(HERE / "training_response_ratios.csv", index=False)
    params.to_csv(HERE / "hill_parameters.csv", index=False)
    curves.to_csv(HERE / "alpha_curves.csv", index=False)
    plot_alpha(ratios, curves, fig_dir)
    print("[4/7] Decoding original and improved predictions", flush=True)
    predicted = predict_both_methods(model, train, latent, hill_models)
    print("[5/7] Evaluating identical held-out A549 conditions", flush=True)
    metrics, gene_means, comparison = evaluate(dataset, predicted)
    metrics.to_csv(HERE / "metrics.csv", index=False)
    comparison.to_csv(HERE / "metrics_comparison.csv", index=False)
    gene_means.to_csv(HERE / "per_gene_mean_expression.csv.gz", index=False, compression="gzip")
    plot_metric_comparison(metrics, fig_dir)
    selected_genes = plot_gene_curves(gene_means, fig_dir)
    selected_genes.to_csv(HERE / "representative_genes.csv", index=False)
    print("[6/7] Building shared UMAP and result AnnData", flush=True)
    result = build_results(dataset, predicted)
    result.write_h5ad(HERE / "prediction_results_task2.h5ad", compression="gzip")
    coords = plot_umap(result, fig_dir)
    coords.to_csv(HERE / "umap_coordinates.csv", index_label="cell_id")
    runtime = time.time() - started
    config = {
        "seed": SEED,
        "target_cell_line": TARGET,
        "drugs": DRUGS,
        "doses_nM": DOSES,
        "original_method": "log1p(dose) / log1p(max_dose)",
        "improved_method": "drug-specific normalized Hill fitted on K562 and MCF7 latent response projections",
        "task1_model_read_only": str(args.model),
        "runtime_seconds": runtime,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    (HERE / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest()
    write_report(metrics, comparison, params, selected_genes, runtime)
    print(f"[7/7] Complete in {runtime / 60:.1f} min: {HERE}", flush=True)


if __name__ == "__main__":
    main()
