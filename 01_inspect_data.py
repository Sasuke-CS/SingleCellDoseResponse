import scanpy as sc
import pandas as pd

DATA_PATH = r"data/SrivatsanTrapnell2020_sciplex3.h5ad"

# backed='r'：先只读元信息，避免一上来把 2.5 GB 全塞进内存
adata = sc.read_h5ad(DATA_PATH, backed="r")

print("=" * 80)
print("AnnData:")
print(adata)

print("\nobs columns:")
print(adata.obs.columns.tolist())

print("\nshape:")
print(adata.shape)

# 重点字段
for col in ["cell_line", "perturbation", "dose_value", "dose_unit", "time"]:
    if col in adata.obs.columns:
        print(f"\n===== {col} =====")
        print(adata.obs[col].value_counts(dropna=False).head(30))

# 三个细胞系
if "cell_line" in adata.obs.columns:
    print("\n===== cell lines =====")
    print(adata.obs["cell_line"].value_counts())

# 检查 Belinostat
if "perturbation" in adata.obs.columns:
    mask = (
        adata.obs["perturbation"]
        .astype(str)
        .str.lower()
        .eq("belinostat")
    )

    bel = adata.obs.loc[mask].copy()

    print("\n===== Belinostat total cells =====")
    print(len(bel))

    group_cols = [
        c for c in
        ["cell_line", "perturbation", "dose_value", "dose_unit", "time"]
        if c in bel.columns
    ]

    print("\n===== Belinostat cell counts =====")
    print(
        bel.groupby(group_cols, observed=True)
           .size()
           .rename("n_cells")
           .reset_index()
           .to_string(index=False)
    )

# control
if "perturbation" in adata.obs.columns:
    ctrl = adata.obs[
        adata.obs["perturbation"].astype(str).str.lower().eq("control")
    ]

    print("\n===== control cells =====")

    group_cols = [
        c for c in ["cell_line", "dose_value", "dose_unit", "time"]
        if c in ctrl.columns
    ]

    print(
        ctrl.groupby(group_cols, observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
            .to_string(index=False)
    )