#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task 1: small-scale scVIDR perturbation-response prediction on sci-Plex 3.

The script intentionally uses a bounded, reproducible subset of the 2.5 GB input,
but keeps every held-out A549 test cell. It writes all tabular results, figures,
predictions, a trained model and a Markdown report beneath this script's directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = HERE / ".runtime_cache"
RUNTIME.mkdir(exist_ok=True)
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
from scipy import sparse, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# The upstream repository uses local (non-package-relative) imports.
sys.path.insert(0, str(ROOT / "vidr"))
import vidr as vidr_module  # noqa: E402
from vidr import VIDR  # noqa: E402
from scvi.data import setup_anndata  # noqa: E402

# Upstream vidr.py calls LinearRegression without importing it (repository bug).
# Injecting it here keeps the experiment self-contained without altering library code.
vidr_module.LinearRegression = LinearRegression


SEED = 20260913
TARGET = "A549"
DRUGS = ["(+)-JQ1", "Trametinib (GSK1120212)"]
DOSES = [10.0, 100.0, 1000.0, 10000.0]
TIME_H = 24.0
CONTROL = "control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "SrivatsanTrapnell2020_sciplex3.h5ad",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--train-cap", type=int, default=300)
    parser.add_argument("--control-cap", type=int, default=500)
    parser.add_argument("--reuse-preprocessed", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild README.md from completed CSV outputs")
    return parser.parse_args()


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def condition_name(perturbation: str, dose: float) -> str:
    if perturbation == CONTROL:
        return "control_0"
    return f"{perturbation}_{dose:g}nM"


def choose_rows(obs: pd.DataFrame, train_cap: int, control_cap: int) -> tuple[np.ndarray, pd.DataFrame]:
    relevant = (
        obs["cell_line"].isin(["A549", "K562", "MCF7"])
        & (obs["time"] == TIME_H)
        & (
            ((obs["perturbation"] == CONTROL) & (obs["dose_value"] == 0))
            | (obs["perturbation"].isin(DRUGS) & obs["dose_value"].isin(DOSES))
        )
    )
    meta = obs.loc[relevant, ["cell_line", "perturbation", "dose_value", "dose_unit", "time"]].copy()
    meta["source_row"] = np.flatnonzero(relevant)
    meta["split"] = np.where(
        (meta["cell_line"] == TARGET) & (meta["perturbation"] != CONTROL), "test", "train"
    )
    rng = np.random.default_rng(SEED)
    keep = []
    for (_, perturbation, _), group in meta.groupby(
        ["cell_line", "perturbation", "dose_value"], sort=True, observed=True
    ):
        if group["split"].iat[0] == "test":
            # Evaluation retains all held-out target cells.
            chosen = group.index.to_numpy()
        else:
            cap = control_cap if perturbation == CONTROL else train_cap
            chosen = rng.choice(group.index.to_numpy(), size=min(cap, len(group)), replace=False)
        keep.extend(chosen.tolist())
    selected = meta.loc[sorted(keep)].copy()
    selected["condition"] = [
        condition_name(p, d) for p, d in zip(selected["perturbation"], selected["dose_value"])
    ]
    return selected["source_row"].to_numpy(), selected


def preprocess(input_path: Path, args: argparse.Namespace, cache_path: Path) -> ad.AnnData:
    print(f"[1/6] Opening {input_path}", flush=True)
    backed = sc.read_h5ad(input_path, backed="r")
    rows, selected = choose_rows(backed.obs, args.train_cap, args.control_cap)
    print(f"      Loading {len(rows):,} selected cells from {backed.n_obs:,}", flush=True)
    dataset = backed[rows, :].to_memory()
    backed.file.close()

    # Retain only the explicit columns used downstream and overwrite with the
    # selection table to preserve an auditable source-row reference.
    selected.index = dataset.obs_names
    dataset.obs = selected
    dataset.var_names_make_unique()
    # Feature selection is fitted on training cells only to avoid test leakage.
    train_mask = dataset.obs["split"].to_numpy() == "train"
    train_x = dataset.X[train_mask]
    if sparse.issparse(train_x):
        expressed_cells = np.asarray(train_x.getnnz(axis=0)).reshape(-1)
    else:
        expressed_cells = np.count_nonzero(train_x, axis=0)
    dataset = dataset[:, expressed_cells >= 20].copy()
    dataset.layers["counts"] = dataset.X.copy()
    sc.pp.normalize_total(dataset, target_sum=1e4)
    sc.pp.log1p(dataset)
    hvg_fit = dataset[train_mask].copy()
    sc.pp.highly_variable_genes(hvg_fit, n_top_genes=min(args.n_hvg, dataset.n_vars), flavor="seurat")
    for field in ["means", "dispersions", "dispersions_norm"]:
        dataset.var[field] = hvg_fit.var[field].to_numpy()
    dispersion_all = dataset.var["dispersions_norm"].fillna(-np.inf).to_numpy()
    exact_hvg = np.argsort(-dispersion_all)[: min(args.n_hvg, dataset.n_vars)]
    keep_hvg = np.zeros(dataset.n_vars, dtype=bool)
    keep_hvg[exact_hvg] = True
    dataset.var["highly_variable"] = keep_hvg
    dataset = dataset[:, keep_hvg].copy()
    # Record rank by normalized dispersion for the Top-500-HVG evaluation.
    dispersion = dataset.var["dispersions_norm"].fillna(-np.inf).to_numpy()
    order = np.argsort(-dispersion)
    rank = np.empty(dataset.n_vars, dtype=int)
    rank[order] = np.arange(1, dataset.n_vars + 1)
    dataset.var["hvg_rank"] = rank
    dataset.uns["task1_config"] = {
        "seed": SEED,
        "target_cell_line": TARGET,
        "drugs": DRUGS,
        "doses_nM": DOSES,
        "time_h": TIME_H,
        "train_cap": args.train_cap,
        "control_cap": args.control_cap,
        "normalization": "library size 1e4, log1p, Seurat HVG",
    }
    dataset.write_h5ad(cache_path, compression="gzip")
    print(f"      Saved {dataset.n_obs:,} cells x {dataset.n_vars:,} genes to {cache_path.name}", flush=True)
    return dataset


def split_count_table(dataset: ad.AnnData) -> pd.DataFrame:
    return (
        dataset.obs.groupby(
            ["split", "cell_line", "perturbation", "dose_value", "time"], observed=True
        )
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["split", "perturbation", "cell_line", "dose_value"])
    )


def train_model(dataset: ad.AnnData, epochs: int, model_dir: Path) -> tuple[VIDR, ad.AnnData]:
    print("[2/6] Registering training split and fitting scVIDR VAE", flush=True)
    train = dataset[dataset.obs["split"] == "train"].copy()
    train.obs["condition"] = train.obs["condition"].astype(str).astype("category")
    train.obs["cell_line"] = train.obs["cell_line"].astype(str).astype("category")
    train = setup_anndata(train, copy=True, batch_key="condition", labels_key="cell_line")
    model = VIDR(
        train,
        hidden_dim=256,
        latent_dim=32,
        n_hidden_layers=2,
        dropout_rate=0.15,
        linear_decoder=False,
    )
    model.train(
        max_epochs=epochs,
        use_gpu=torch.cuda.is_available(),
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=15,
        check_val_every_n_epoch=1,
    )
    model.save(model_dir, overwrite=True)
    history_parts = []
    for key, values in model.history.items():
        frame = pd.DataFrame({key: np.asarray(values).reshape(-1)})
        frame["epoch"] = np.arange(1, len(frame) + 1)
        history_parts.append(frame.set_index("epoch"))
    if history_parts:
        pd.concat(history_parts, axis=1).to_csv(HERE / "training_history.csv", index_label="epoch")
    return model, train


def predict_all(model: VIDR, dataset: ad.AnnData) -> ad.AnnData:
    print("[3/6] Predicting eight held-out drug-dose conditions", flush=True)
    pieces = []
    ctrl_key = condition_name(CONTROL, 0)
    for drug in DRUGS:
        for dose in DOSES:
            treat_key = condition_name(drug, dose)
            pred, delta, reg = model.predict(
                ctrl_key=ctrl_key,
                treat_key=treat_key,
                cell_type_to_predict=TARGET,
                regression=True,
                continuous=False,
            )
            pred.obs = pred.obs[["cell_line"]].copy()
            pred.obs["cell_line"] = TARGET
            pred.obs["perturbation"] = drug
            pred.obs["dose_value"] = dose
            pred.obs["dose_unit"] = "nM"
            pred.obs["time"] = TIME_H
            pred.obs["condition"] = treat_key
            pred.obs["source"] = "predicted"
            pred.uns["predicted_delta"] = np.asarray(delta)
            pred.obs_names = [f"pred_{drug}_{dose:g}_{i}" for i in range(pred.n_obs)]
            pieces.append(pred)
            print(f"      {drug}, {dose:g} nM: {pred.n_obs} predicted cells", flush=True)

    predicted = ad.concat(pieces, join="inner", merge="same", index_unique=None)
    return predicted


def dense_mean(x) -> np.ndarray:
    result = x.mean(axis=0)
    return np.asarray(result).reshape(-1)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.pearsonr(x, y)[0])


def evaluate(dataset: ad.AnnData, predicted: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[4/6] Computing Pearson correlation and R2", flush=True)
    ctrl = dataset[(dataset.obs["cell_line"] == TARGET) & (dataset.obs["perturbation"] == CONTROL)]
    ctrl_mean = dense_mean(ctrl.X)
    hvg500 = dataset.var["hvg_rank"].to_numpy() <= min(500, dataset.n_vars)
    rows = []
    gene_rows = []
    genes = dataset.var_names.to_numpy()
    for drug in DRUGS:
        for dose in DOSES:
            truth = dataset[
                (dataset.obs["cell_line"] == TARGET)
                & (dataset.obs["perturbation"] == drug)
                & (dataset.obs["dose_value"] == dose)
            ]
            pred = predicted[
                (predicted.obs["perturbation"] == drug)
                & (predicted.obs["dose_value"] == dose)
            ]
            true_mean = dense_mean(truth.X)
            pred_mean = dense_mean(pred.X)
            de_order = np.argsort(-np.abs(true_mean - ctrl_mean))
            de100 = np.zeros(dataset.n_vars, dtype=bool)
            de100[de_order[: min(100, dataset.n_vars)]] = True
            sets = {
                "all_retained_genes": np.ones(dataset.n_vars, dtype=bool),
                "top500_HVG": hvg500,
                "top100_DE": de100,
            }
            for gene_set, mask in sets.items():
                rows.append(
                    {
                        "drug": drug,
                        "dose_nM": dose,
                        "gene_set": gene_set,
                        "n_genes": int(mask.sum()),
                        "n_true_cells": truth.n_obs,
                        "n_pred_cells": pred.n_obs,
                        "pearson": safe_pearson(true_mean[mask], pred_mean[mask]),
                        "r2": float(r2_score(true_mean[mask], pred_mean[mask])),
                        "control_baseline_pearson": safe_pearson(true_mean[mask], ctrl_mean[mask]),
                        "control_baseline_r2": float(r2_score(true_mean[mask], ctrl_mean[mask])),
                    }
                )
            for i, gene in enumerate(genes):
                gene_rows.append(
                    {
                        "drug": drug,
                        "dose_nM": dose,
                        "gene": gene,
                        "true_mean": true_mean[i],
                        "predicted_mean": pred_mean[i],
                        "control_mean": ctrl_mean[i],
                        "hvg_rank": int(dataset.var["hvg_rank"].iat[i]),
                        "is_top100_DE": bool(de100[i]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(gene_rows)


def build_result_adata(dataset: ad.AnnData, predicted: ad.AnnData) -> ad.AnnData:
    ctrl = dataset[(dataset.obs["cell_line"] == TARGET) & (dataset.obs["perturbation"] == CONTROL)].copy()
    ctrl.obs["source"] = "control"
    truth = dataset[dataset.obs["split"] == "test"].copy()
    truth.obs["source"] = "true"
    result = ad.concat([ctrl, truth, predicted], join="inner", merge="same", index_unique="-")
    for col in ["cell_line", "perturbation", "condition", "source"]:
        result.obs[col] = result.obs[col].astype(str).astype("category")
    return result


def plot_metrics(metrics: pd.DataFrame, fig_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    all_genes = metrics[metrics["gene_set"] == "all_retained_genes"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for drug, group in all_genes.groupby("drug", sort=False):
        label = drug.replace(" (GSK1120212)", "")
        axes[0].plot(group["dose_nM"], group["pearson"], marker="o", label=label)
        axes[0].plot(group["dose_nM"], group["control_baseline_pearson"], marker="x", linestyle="--", alpha=.65)
        axes[1].plot(group["dose_nM"], group["r2"], marker="o", label=label)
        axes[1].plot(group["dose_nM"], group["control_baseline_r2"], marker="x", linestyle="--", alpha=.65)
    axes[0].set_ylabel("Pearson correlation")
    axes[1].set_ylabel("R²")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Dose (nM, log scale)")
        ax.axhline(0, color="black", linewidth=.7)
    axes[0].legend(title="Solid: scVIDR\nDashed: control")
    fig.suptitle("Held-out A549 mean-expression prediction (all retained genes)")
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_by_dose.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = all_genes["drug"].str.replace(" (GSK1120212)", "", regex=False) + "\n" + all_genes["dose_nM"].map("{:g} nM".format)
    for ax, field, title in zip(axes, ["pearson", "r2"], ["Pearson", "R²"]):
        sub = metrics.pivot_table(index=["drug", "dose_nM"], columns="gene_set", values=field).loc[
            list(zip(all_genes["drug"], all_genes["dose_nM"]))
        ]
        sub.index = labels
        sns.heatmap(sub, annot=True, fmt=".3f", cmap="vlag", center=0 if field == "r2" else None, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fig_dir / "metric_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_umap(result: ad.AnnData, fig_dir: Path) -> pd.DataFrame:
    print("[5/6] Building a shared PCA/UMAP space", flush=True)
    rng = np.random.default_rng(SEED)
    chosen = []
    group_cols = ["source", "perturbation", "dose_value"]
    for _, frame in result.obs.groupby(group_cols, observed=True):
        idx = result.obs_names.get_indexer(frame.index)
        chosen.extend(rng.choice(idx, size=min(150, len(idx)), replace=False).tolist())
    vis = result[np.asarray(chosen)].copy()
    vis.X = vis.X.astype(np.float32)
    # PCA is fitted directly on the common log-normalized expression matrix.
    # Avoiding a second gene-wise scaling prevents low-variance decoder artifacts
    # from dominating distances between raw and decoded cells.
    sc.tl.pca(vis, n_comps=30, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(vis, n_neighbors=20, n_pcs=30, random_state=SEED)
    sc.tl.umap(vis, random_state=SEED, min_dist=0.35)
    coords = pd.DataFrame(vis.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=vis.obs_names)
    coords = pd.concat([coords, vis.obs.reset_index(drop=True).set_index(coords.index)], axis=1)

    fig, axes = plt.subplots(len(DRUGS), len(DOSES), figsize=(16, 7), sharex=True, sharey=True)
    colors = {"control": "#808080", "true": "#377eb8", "predicted": "#e41a1c"}
    for i, drug in enumerate(DRUGS):
        for j, dose in enumerate(DOSES):
            ax = axes[i, j]
            mask = (coords["source"] == "control") | (
                (coords["perturbation"] == drug) & (coords["dose_value"] == dose)
            )
            frame = coords.loc[mask]
            for source in ["control", "true", "predicted"]:
                s = frame[frame["source"] == source]
                ax.scatter(s["UMAP1"], s["UMAP2"], s=7, alpha=.55, c=colors[source], label=source, rasterized=True)
            if i == 0:
                ax.set_title(f"{dose:g} nM")
            if j == 0:
                ax.set_ylabel(drug.replace(" (GSK1120212)", "") + "\nUMAP2")
            if i == len(DRUGS) - 1:
                ax.set_xlabel("UMAP1")
            ax.set_xticks([])
            ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.suptitle("A549 control, held-out truth and scVIDR predictions in one shared UMAP", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(fig_dir / "umap_control_true_predicted.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return coords


def make_report(metrics: pd.DataFrame, dataset: ad.AnnData, runtime_s: float, args: argparse.Namespace) -> None:
    all_metrics = metrics[metrics["gene_set"] == "all_retained_genes"].copy()
    summary = all_metrics.groupby("drug")[["pearson", "r2", "control_baseline_pearson", "control_baseline_r2"]].mean()
    all_metrics["r2_gain_over_control"] = all_metrics["r2"] - all_metrics["control_baseline_r2"]
    improved_n = int((all_metrics["r2_gain_over_control"] > 0).sum())
    mean_gain = float(all_metrics["r2_gain_over_control"].mean())
    best = all_metrics.loc[all_metrics["r2"].idxmax()]
    worst = all_metrics.loc[all_metrics["r2"].idxmin()]
    de_metrics = metrics[metrics["gene_set"] == "top100_DE"]
    table = all_metrics[["drug", "dose_nM", "pearson", "r2", "control_baseline_pearson", "control_baseline_r2"]].copy()
    for col in table.columns[2:]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    def markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
        """Small dependency-free Markdown table writer (tabulate is not installed)."""
        shown = frame.copy()
        if include_index:
            shown.insert(0, shown.index.name or "drug", shown.index.astype(str))
        headers = [str(x) for x in shown.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in shown.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
        return "\n".join(lines)

    table_md = markdown_table(table)
    summary_md = markdown_table(summary.round(4), include_index=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    report = f"""# 任务一：scVIDR 单细胞药物扰动响应预测

## 结论摘要

本实验以 **A549** 为完全留出的目标细胞系，测试 **(+)-JQ1** 和 **Trametinib (GSK1120212)** 在 10、100、1000、10000 nM（24 h）下的响应。模型未看到任何 A549 的测试药物非零剂量细胞。八个条件在全部 {dataset.n_vars:,} 个保留基因上的结果如下。

{table_md}

按药物跨剂量平均：

{summary_md}

最高 R² 条件为 **{best['drug']} / {best['dose_nM']:g} nM**（R²={best['r2']:.4f}）；最低为 **{worst['drug']} / {worst['dose_nM']:g} nM**（R²={worst['r2']:.4f}）。CSV 同时给出“直接使用 A549 对照均值”的朴素基线。scVIDR 在 {improved_n}/8 个条件上提高 R²，平均变化为 {mean_gain:+.4f}；这说明总体跨基因相关性虽高，但其中一部分来自稳定的细胞系表达背景，模型是否恢复扰动必须结合基线、DE 基因指标和 UMAP 判断。

![各剂量指标](figures/metrics_by_dose.png)

![指标热图](figures/metric_heatmaps.png)

Top 100 DE 基因比全基因更严格：其 Pearson 范围为 {de_metrics['pearson'].min():.4f}–{de_metrics['pearson'].max():.4f}，R² 范围为 {de_metrics['r2'].min():.4f}–{de_metrics['r2'].max():.4f}。Trametinib 的 DE 指标明显低于其全基因指标，说明模型较好保留了整体表达背景，但对强扰动基因的幅度恢复仍有限。

## scVIDR 原理

scVIDR 先用变分自编码器（VAE）将高维基因表达编码到低维潜在空间。对于每个药物—剂量条件，在非目标细胞系中计算处理与对照潜在中心之差（扰动向量）；随后用对照潜在中心到扰动向量的线性回归，把跨细胞系观察到的响应外推到目标 A549。最后将 A549 对照细胞的潜在表示加上预测扰动向量并经解码器还原为基因表达。这里每个剂量独立估计方向，属于任务一的条件级扰动预测，不使用任务二的剂量插值。

## 数据预处理与划分

- 原始文件：`data/SrivatsanTrapnell2020_sciplex3.h5ad`（799,317 个细胞，110,983 个原始特征）。
- 选择依据：两个药物在 A549、K562、MCF7 的 24 h 数据中均具备四个非零剂量，且每个目标测试条件有 167–223 个细胞；满足至少三个剂量且细胞数充足的要求。
- 训练集：A549/K562/MCF7 的 0 nM 对照（每系固定随机抽样至 {args.control_cap} 个），以及 K562、MCF7 中两个测试药物的全部四剂量（每组合至多 {args.train_cap} 个）。
- 测试集：A549 中两个测试药物的全部四剂量细胞，不抽样，共 {int((dataset.obs['split'] == 'test').sum()):,} 个。
- 预处理：仅使用训练细胞拟合过滤与 HVG 选择（避免测试泄漏）；基因至少在 20 个训练细胞表达；所有细胞按相同规则将总量归一化至 10,000 并 `log1p`；再按训练集 Seurat dispersion 精确保留 {dataset.n_vars:,} 个 HVG。VAE 为 2 层、256 隐单元、32 维潜变量，最多 {args.epochs} epochs，early stopping patience=15，随机种子 {SEED}。
- 硬件：{gpu}。总脚本运行时间约 {runtime_s / 60:.1f} 分钟。

详细实际细胞数见 `split_counts.csv`，选定基因及 HVG 排名见 `selected_genes.csv`。

## 评价定义与解读

每个测试条件分别计算真实细胞与预测细胞的基因均值。`Pearson` 衡量跨基因模式的一致性；`R² = 1 - SSE/SST` 衡量绝对预测误差，因此允许为负值。分别评价全部保留基因、Top 500 HVG 和按该条件真实处理均值相对真实对照均值的绝对差选出的 Top 100 DE 基因。Top DE 是评价集合而非训练特征，使用真实测试标签选基因会带来评价集合选择上的信息使用，故不能视作完全无监督选基因；报告同时保留全基因和 HVG 指标作为主结果。

UMAP 将 A549 对照、真实处理和预测细胞合并，统一使用前述 log-normalized 表达矩阵，并在同一 PCA、邻接图和 UMAP 映射中计算，未分别拟合坐标。JQ1 的预测与真实细胞整体重叠较好；Trametinib 的预测覆盖了真实区域但更分散，提示单细胞分布的拟合弱于均值指标。

![共享 UMAP](figures/umap_control_true_predicted.png)

## 本任务中的稳健性改进

相对仓库示例流程，本实现增加了四点：HVG/基因过滤只在训练细胞上拟合以避免测试泄漏；固定随机种子并记录原始行号与实际划分；使用 early stopping 限制过拟合；为每个指标提供对照均值基线并补充 DE 基因评价，避免只报告容易被稳定背景抬高的全基因相关性。上游 `vidr.py` 漏导入 `LinearRegression` 的兼容问题也在本脚本内局部修复，未修改库源码。

## 结果文件说明

- `metrics.csv`：8 个条件 × 3 个基因集合的 Pearson、R² 与对照基线。
- `per_gene_mean_expression.csv.gz`：逐条件逐基因的真实、预测、对照均值及 DE/HVG 标记。
- `prediction_results.h5ad`：A549 对照、真实测试细胞及模型预测细胞（统一基因空间）。
- `umap_coordinates.csv`：绘图用共享 UMAP 坐标和细胞标签。
- `task1_preprocessed.h5ad`：可复用的归一化/HVG 后数据子集，含 `counts` 层。
- `model/`：已训练 scVIDR 模型；`training_history.csv` 为训练轨迹。
- `figures/`：指标曲线、指标热图和共享 UMAP。
- `run_task1.py`：从原始 h5ad 重新生成全部主要结果的入口。
- `run_config.json`：运行参数、软件版本和耗时。

## 局限与可改进点

数据只有两个非目标细胞系可供 scVIDR 的跨细胞系回归，回归样本数明显少于 32 维潜变量，外推稳定性受限。当前小规模实验还只纳入两个测试药物，VAE 对更广药物状态的表示能力有限。后续可增加训练药物、对潜在回归使用岭正则/降维、进行多随机种子与 bootstrap 置信区间，并与 pooled-delta、scGen、对照均值等基线系统比较。任务二还应在相同留出集上显式比较原始 log-linear 剂量插值和改进缩放函数。

## 复现

在仓库根目录执行：

```powershell
conda activate scVIDR
python "任务一/run_task1.py"
```

若希望复用已生成的预处理数据、仅重新训练与评估，增加 `--reuse-preprocessed`。固定随机种子降低随机性，但 GPU 算子和旧版 PyTorch/scvi-tools 仍可能造成细微数值差异。
"""
    (HERE / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    seed_everything()
    fig_dir = HERE / "figures"
    model_dir = HERE / "model"
    fig_dir.mkdir(exist_ok=True)
    cache_path = HERE / "task1_preprocessed.h5ad"
    if args.report_only:
        dataset = sc.read_h5ad(cache_path)
        metrics = pd.read_csv(HERE / "metrics.csv")
        config_path = HERE / "run_config.json"
        runtime = json.loads(config_path.read_text(encoding="utf-8")).get("runtime_seconds", 0.0)
        make_report(metrics, dataset, runtime, args)
        print(f"Rebuilt {HERE / 'README.md'}", flush=True)
        return
    if args.reuse_preprocessed and cache_path.exists():
        print(f"[1/6] Reusing {cache_path.name}", flush=True)
        dataset = sc.read_h5ad(cache_path)
    else:
        dataset = preprocess(args.input, args, cache_path)

    counts = split_count_table(dataset)
    counts.to_csv(HERE / "split_counts.csv", index=False)
    dataset.var[["highly_variable", "means", "dispersions", "dispersions_norm", "hvg_rank"]].to_csv(
        HERE / "selected_genes.csv", index_label="gene"
    )
    model, train = train_model(dataset, args.epochs, model_dir)
    predicted = predict_all(model, dataset)
    metrics, gene_means = evaluate(dataset, predicted)
    metrics.to_csv(HERE / "metrics.csv", index=False)
    gene_means.to_csv(HERE / "per_gene_mean_expression.csv.gz", index=False, compression="gzip")
    result = build_result_adata(dataset, predicted)
    result.write_h5ad(HERE / "prediction_results.h5ad", compression="gzip")
    plot_metrics(metrics, fig_dir)
    coords = plot_umap(result, fig_dir)
    coords.to_csv(HERE / "umap_coordinates.csv", index_label="cell_id")

    runtime = time.time() - started
    config = {
        "seed": SEED,
        "target_cell_line": TARGET,
        "drugs": DRUGS,
        "doses_nM": DOSES,
        "time_h": TIME_H,
        "epochs_requested": args.epochs,
        "n_hvg": dataset.n_vars,
        "train_cap": args.train_cap,
        "control_cap": args.control_cap,
        "runtime_seconds": runtime,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "versions": {
            "python": sys.version.split()[0],
            "scanpy": sc.__version__,
            "anndata": ad.__version__,
            "torch": torch.__version__,
        },
    }
    (HERE / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(metrics, dataset, runtime, args)
    print(f"[6/6] Complete in {runtime / 60:.1f} min. Results: {HERE}", flush=True)


if __name__ == "__main__":
    main()
