由 Omar Kana 开发 <kanaomar@msu.edu>

CLI 集成：David Filipovic <filipov4@msu.edu>

维护者：David Filipovic <filipov4@msu.edu>

# scVIDR

单细胞剂量反应变分推断（Single Cell Variational Inference of the Dose Response，scVIDR）是一个变分自编码器工具，用于预测不同细胞类型对化学扰动的表达响应。

[![DOI](https://zenodo.org/badge/549268647.svg)](https://zenodo.org/badge/latestdoi/549268647)

## 论文

[发表于 Patterns](https://doi.org/10.1016/j.patter.2023.100817)

## 安装

```bash
git clone https://github.com/BhattacharyaLab/scVIDR.git
cd scVIDR
conda create -n scVIDR python=3.8.5
conda activate scVIDR
pip3 install -r requirements.txt
pip3 install geomloss==0.2.5
pip install torch==1.8.1+cu111 torchaudio==0.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## 数据

图表 Notebook 所需的数据目录可从以下地址获取：

https://drive.google.com/file/d/11fzDbp0B19Dy47MtD742Jl4Hz2bdSiiq/view?usp=sharing

下载 `data.zip` 后，将其复制到 `scVIDR/data` 目录并在那里解压。

## VAE 模型训练（单剂量和多剂量模型）

```text
运行 `scvidr_train.py single_dose` 命令训练单剂量 VAE 模型（论文图 2）。

运行 `scvidr_train.py multi_dose` 命令训练多剂量 VAE 模型（论文图 3）。
```

两种模型都要求输入格式为 h5ad 的 AnnData 文件。除单细胞 RNA 测序数据外，`obs` 表中还必须包含剂量列和细胞类型列。
单剂量模型至少需要两个不同的剂量，多剂量模型至少需要三个不同的剂量。

两种模型的命令参数相同，具体如下。

用法：

```text
scvidr_train.py {single_dose/multi_dose} [-h] [--dose_column DOSE_COLUMN] [--celltype_column CELLTYPE_COLUMN] [--test_celltype TEST_CELLTYPE] [--treated_dose CONTROL_DOSE] [--treated_dose TREATED_DOSE] [--celltypes_keep CELLTYPES_KEEP] h5ad_data_file model_path
```

使用 h5ad 输入数据集训练可用于 scGen 和 scVIDR 的 VAE 模型。

位置参数：

```text
  h5ad_data_file        h5ad 格式的原始读数数据文件

  model_path            训练好的模型保存目录
```

模型参数：

```text
  -h, --help
                        显示帮助信息
  --dose_column DOSE_COLUMN
                        obs 数据框中表示剂量的列名（默认值为 "Dose"）
  --celltype_column CELLTYPE_COLUMN
                        obs 数据框中表示细胞类型的列名（默认值为 "celltype"）
  --test_celltype TEST_CELLTYPE
                        留出用于测试的细胞类型名称；包含空格的细胞类型请使用引号括起来
                        （默认值为 "Hepatocytes - portal"）
  --control_dose CONTROL_DOSE
                        对照剂量（默认值为 "0"）
  --treated_dose TREATED_DOSE
                        处理剂量（默认值为 "30"）
  --celltypes_keep CELLTYPES_KEEP
                        训练/测试时保留的数据集中的细胞类型。可以提供一个文件（每行一个细胞类型），
                        或使用分号分隔的细胞类型列表（包含空格的细胞类型请使用引号括起来）。
                        默认保留所有可用的细胞类型（默认值为 "ALL"）
```

执行以下命令，可训练论文中使用的全部单细胞类型单剂量模型：

```bash
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - central.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - central" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Cholangiocytes.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Stellate Cells.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/"
python scvidr_train.py single_dose --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_singleDose.h5ad "../data/VAE_Binary_Prediction_Dioxin_5000g_Endothelial Cells.pt/"
```

执行以下命令，可训练论文中使用的全部单细胞类型多剂量模型：

```bash
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - central" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes - central.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Hepatocytes - portal" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes - portal.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Cholangiocytes" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Cholangiocytes.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Stellate Cells" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Stellate Cells.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Portal Fibroblasts" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Portal Fibroblasts.pt/"
python scvidr_train.py multi_dose --control_dose 0.0 --celltypes_keep ../metadata/liver_celltypes --test_celltype "Endothelial Cells" ../data/nault2021_multiDose.h5ad "../data/VAE_Cont_Prediction_Dioxin_5000g_Endothelial Cells.pt/"
```

**注意**：以上模型均提供预训练版本，名称与上面列出的名称相同。

## 单剂量模型预测

```text
用法：scvidr_predict.py single_dose [-h] [--model MODEL]
                                     [--dose_column DOSE_COLUMN]
                                     [--celltype_column CELLTYPE_COLUMN]
                                     [--test_celltype TEST_CELLTYPE]
                                     [--control_dose CONTROL_DOSE]
                                     [--treated_dose TREATED_DOSE]
                                     [--celltypes_keep CELLTYPES_KEEP]
                                     h5ad_data_file model_path output_path
```

使用预训练的 scVIDR 或 scGen 模型预测处理条件。

位置参数：

```text
h5ad_data_file        h5ad 格式的原始读数数据文件
model_path            训练模型步骤中保存的模型目录
output_path           AnnData 输出目录，结果将保存为 h5ad 格式
```

可选参数：

```text
  -h, --help
                        显示帮助信息
  --model MODEL
                        使用 scVIDR 或 scGen 进行预测（默认值为 "scVIDR"）
  --dose_column DOSE_COLUMN
                        obs 数据框中表示剂量的列名（默认值为 "Dose"）
  --celltype_column CELLTYPE_COLUMN
                        obs 数据框中表示细胞类型的列名（默认值为 "celltype"）
  --test_celltype TEST_CELLTYPE
                        留出用于测试的细胞类型名称；包含空格的细胞类型请使用引号括起来
                        （默认值为 "Hepatocytes - portal"）
  --control_dose CONTROL_DOSE
                        对照剂量（默认值为 "0"）
  --treated_dose TREATED_DOSE
                        处理剂量（默认值为 "30"）
  --celltypes_keep CELLTYPES_KEEP
                        训练/测试时保留的数据集中的细胞类型。可以提供一个文件（每行一个细胞类型），
                        或使用分号分隔的细胞类型列表；包含空格的细胞类型请使用引号括起来。
                        默认保留所有可用的细胞类型（默认值为 "ALL"）
```

单剂量预测命令示例：

```bash
python scvidr_predict.py single_dose ../data/nault2021_singleDose.h5ad ../data/VAE_Binary_Prediction_Dioxin_5000g_Hepatocytes\ -\ portal.pt/ ../data/SingleDose_TCDD --model scVIDR --dose_column Dose --celltype_column celltype --test_celltype "Hepatocytes - portal" --control_dose 0 --treated_dose 30 --celltypes_keep ../metadata/liver_celltypes
```

## 多剂量模型预测

```text
用法：scvidr_predict.py multi_dose [-h] [--model MODEL]
                                    [--dose_column DOSE_COLUMN]
                                    [--celltype_column CELLTYPE_COLUMN]
                                    [--test_celltype TEST_CELLTYPE]
                                    [--control_dose CONTROL_DOSE]
                                    [--treated_dose TREATED_DOSE]
                                    [--celltypes_keep CELLTYPES_KEEP]
                                    h5ad_data_file model_path output_path
```

位置参数：

```text
h5ad_data_file        h5ad 格式的原始读数数据文件
model_path            训练模型步骤中保存的模型目录
output_path           AnnData 输出目录，结果将保存为 h5ad 格式
```

可选参数：

```text
  -h, --help
                        显示帮助信息
  --model MODEL
                        使用 scVIDR 或 scGen 进行预测（默认值为 "scVIDR"）
  --dose_column DOSE_COLUMN
                        obs 数据框中表示剂量的列名（默认值为 "Dose"）
  --celltype_column CELLTYPE_COLUMN
                        obs 数据框中表示细胞类型的列名（默认值为 "celltype"）
  --test_celltype TEST_CELLTYPE
                        留出用于测试的细胞类型名称；包含空格的细胞类型请使用引号括起来
                        （默认值为 "Hepatocytes - portal"）
  --control_dose CONTROL_DOSE
                        对照剂量（默认值为 "0"）
  --treated_dose TREATED_DOSE
                        处理剂量（默认值为 "30"）
  --celltypes_keep CELLTYPES_KEEP
                        训练/测试时保留的数据集中的细胞类型。可以提供一个文件（每行一个细胞类型），
                        或使用分号分隔的细胞类型列表；包含空格的细胞类型请使用引号括起来。
                        默认保留所有可用的细胞类型（默认值为 "ALL"）
```

多剂量预测命令示例：

```bash
python scvidr_predict.py multi_dose ../data/nault2021_multiDose.h5ad ../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes\ -\ portal.pt/ ../data/MultiDose_TCDD --model scVIDR --dose_column Dose --celltype_column celltype --test_celltype "Hepatocytes - portal" --control_dose 0.0 --treated_dose 30.0 --celltypes_keep ../metadata/liver_celltypes
```

## 计算基因评分

```text
用法：scvidr_genescores.py [-h] [--dose_column DOSE_COLUMN]
                            [--celltype_column CELLTYPE_COLUMN]
                            [--test_celltype TEST_CELLTYPE]
                            [--control_dose CONTROL_DOSE]
                            [--treated_dose TREATED_DOSE]
                            [--celltypes_keep CELLTYPES_KEEP]
                            [--training_size TRAINING_SIZE]
                            h5ad_data_file model_path output_path
```

使用岭回归解释 scVIDR 的预测结果，并输出基因评分 CSV 文件。

位置参数：

```text
h5ad_data_file        h5ad 格式的原始读数数据文件
model_path            训练模型步骤中保存的模型目录
output_path           基因评分 CSV 文件的保存路径
```

可选参数：

```text
  -h, --help
                        显示帮助信息
  --dose_column DOSE_COLUMN
                        obs 数据框中表示剂量的列名（默认值为 "Dose"）
  --celltype_column CELLTYPE_COLUMN
                        obs 数据框中表示细胞类型的列名（默认值为 "celltype"）
  --test_celltype TEST_CELLTYPE
                        留出用于测试的细胞类型名称；包含空格的细胞类型请使用引号括起来
                        （默认值为 "Hepatocytes - portal"）
  --control_dose CONTROL_DOSE
                        对照剂量（默认值为 "0"）
  --treated_dose TREATED_DOSE
                        处理剂量（默认值为 "30"）
  --celltypes_keep CELLTYPES_KEEP
                        训练/测试时保留的数据集中的细胞类型。可以提供一个文件（每行一个细胞类型），
                        或使用分号分隔的细胞类型列表；包含空格的细胞类型请使用引号括起来。
                        默认保留所有可用的细胞类型（默认值为 "ALL"）
  --training_size TRAINING_SIZE
                        从潜在分布中生成的样本数量
```

计算基因评分的示例：

```bash
python scvidr_genescores.py ../data/nault2021_multiDose.h5ad ../data/VAE_Cont_Prediction_Dioxin_5000g_Hepatocytes\ -\ portal.pt/ ../data/MultiDose_TCDD --dose_column Dose --celltype_column celltype --test_celltype "Hepatocytes - portal" --control_dose 0.0 --treated_dose 30.0 --celltypes_keep ../metadata/liver_celltypes
```

## 图表 Notebook

| 图表 | Notebook 路径 | 描述 |
| --- | --- | --- |
| [*图 2*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/Figure2.ipynb) | `notebooks/Figure2.ipynb` | 单剂量 TCDD |
| [*图 3*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/Figure3.ipynb) | `notebooks/Figure3.ipynb` | 多剂量 TCDD |
| [*图 4*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/Figure4.ipynb) | `notebooks/Figure4.ipynb` | 基因评分 |
| [*图 5*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/Figure5.ipynb) | `notebooks/Figure5.ipynb` | 伪剂量 |
| [*补充图 2*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure2.ipynb) | `notebooks/SupplementalFigure2.ipynb` | $\\delta$ 的 PCA |
| [*补充图 3*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure3.ipynb) | `notebooks/SupplementalFigure3.ipynb` | 单剂量 IFNB |
| [*补充图 4*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure4.ipynb) | `notebooks/SupplementalFigure4.ipynb` | 多剂量 sciplex |
| [*补充图 5*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure5.ipynb) | `notebooks/SupplementalFigure5.ipynb` | scVIDR TCDD 分析 |
| [*补充图 6*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure6.ipynb) | `notebooks/SupplementalFigure6.ipynb` | scVIDR sciplex 分析 |
| [*补充图 7*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure7.ipynb) | `notebooks/SupplementalFigure7.ipynb` | scVIDR 跨研究分析 |
| [*补充图 8*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure8.ipynb) | `notebooks/SupplementalFigure8.ipynb` | scVIDR 跨物种分析 |
| [*补充图 9*](https://nbviewer.org/github/BhattacharyaLab/scVIDR/blob/main/notebooks/SupplementalFigure9.ipynb) | `notebooks/SupplementalFigure9.ipynb` | scVIDR 与 scGen 对比 |
