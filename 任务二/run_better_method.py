#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Formal third-method experiment: dose-specific direction regression (DSDR)."""
from __future__ import annotations
import json, os, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;TASK1=ROOT/"任务一";RUNTIME=HERE/".runtime_cache";RUNTIME.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR",str(RUNTIME));os.environ.setdefault("MPLCONFIGDIR",str(RUNTIME));os.environ.setdefault("JOBLIB_TEMP_FOLDER",str(RUNTIME))
import anndata as ad
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,scanpy as sc,seaborn as sns,torch
from scipy import stats
from sklearn.metrics import r2_score
sys.path.insert(0,str(HERE))
from dose_specific_direction import predict_target_delta
from run_task2 import CONTROL,DOSES,DRUGS,SEED,TARGET,condition_name,decode,load_preserved_model


def predict(model,train,latent):
 o=train.obs; pieces=[]
 arrays=(o.cell_line.astype(str).to_numpy(),o.perturbation.astype(str).to_numpy(),o.dose_value.to_numpy())
 for drug in DRUGS:
  for dose in DOSES:
   zc,delta=predict_target_delta(latent,*arrays,drug,dose,TARGET)
   x=decode(model,zc+delta)
   obs=pd.DataFrame({"cell_line":TARGET,"perturbation":drug,"dose_value":dose,"dose_unit":"nM","time":24.,"condition":condition_name(drug,dose),"source":"dose_specific_direction"},index=[f"dsdr_{drug}_{dose:g}_{i}" for i in range(len(x))])
   pieces.append(ad.AnnData(X=x,obs=obs,var=train.var.copy()))
 return ad.concat(pieces,join="inner",merge="same",index_unique=None)


def evaluate(data,pred):
 ctrl=data[(data.obs.cell_line==TARGET)&(data.obs.perturbation==CONTROL)];cm=np.asarray(ctrl.X.mean(axis=0)).reshape(-1);rows=[];gene_rows=[];hvg=data.var.hvg_rank.to_numpy()<=500
 for drug in DRUGS:
  for dose in DOSES:
   truth=data[(data.obs.cell_line==TARGET)&(data.obs.perturbation==drug)&(data.obs.dose_value==dose)];p=pred[(pred.obs.perturbation==drug)&(pred.obs.dose_value==dose)];tm=np.asarray(truth.X.mean(axis=0)).reshape(-1);pm=np.asarray(p.X.mean(axis=0)).reshape(-1);de=np.zeros(data.n_vars,bool);de[np.argsort(-abs(tm-cm))[:100]]=1
   for gs,m in {"all_retained_genes":np.ones(data.n_vars,bool),"top500_HVG":hvg,"top100_DE":de}.items():rows.append({"drug":drug,"dose_nM":dose,"method":"dose_specific_direction","gene_set":gs,"n_genes":int(m.sum()),"pearson":stats.pearsonr(tm[m],pm[m])[0],"r2":r2_score(tm[m],pm[m]),"n_true_cells":truth.n_obs,"n_pred_cells":p.n_obs})
   for i,gene in enumerate(data.var_names):gene_rows.append({"drug":drug,"dose_nM":dose,"gene":gene,"true_mean":tm[i],"predicted_mean":pm[i],"control_mean":cm[i],"is_top100_DE":bool(de[i]),"hvg_rank":int(data.var.hvg_rank.iat[i])})
 return pd.DataFrame(rows),pd.DataFrame(gene_rows)


def comparison_table(combined):
 w=combined.pivot_table(index=["drug","dose_nM","gene_set"],columns="method",values=["pearson","r2"]);w.columns=[f"{metric}_{method}" for metric,method in w.columns];w=w.reset_index()
 for method in ["improved_hill","dose_specific_direction"]:
  w[f"pearson_gain_{method}_vs_log"]=w[f"pearson_{method}"]-w["pearson_original_loglinear"]
  w[f"r2_gain_{method}_vs_log"]=w[f"r2_{method}"]-w["r2_original_loglinear"]
 return w


def plot_metrics(combined,figdir):
 sns.set_theme(style="whitegrid");palette={"original_loglinear":"#377eb8","improved_hill":"#e41a1c","dose_specific_direction":"#4daf4a"};fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
 d=combined[combined.gene_set=="top100_DE"]
 for col,drug in enumerate(DRUGS):
  f=d[d.drug==drug]
  for method,g in f.groupby('method',sort=False):
   label={"original_loglinear":"original log-linear","improved_hill":"Hill","dose_specific_direction":"dose-specific direction"}[method];axes[0,col].plot(g.dose_nM,g.pearson,marker='o',label=label,color=palette[method]);axes[1,col].plot(g.dose_nM,g.r2,marker='o',label=label,color=palette[method])
  axes[0,col].set_title(drug.replace(' (GSK1120212)',''));axes[0,col].set_ylabel('Top 100 DE Pearson');axes[1,col].set_ylabel('Top 100 DE R²');axes[1,col].set_xlabel('Dose (nM, log scale)')
  for row in range(2):axes[row,col].set_xscale('log');axes[row,col].legend(fontsize=8)
 fig.suptitle('Three-method comparison on perturbation-responsive genes');fig.tight_layout();fig.savefig(figdir/'better_method_metrics.png',dpi=220,bbox_inches='tight');plt.close(fig)


def plot_genes(gene_means,old_means,figdir):
 merged=old_means.merge(gene_means,on=['drug','dose_nM','gene'],suffixes=('_old','_dsdr'));chosen=[];fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for ax,drug in zip(axes,DRUGS):
  f=merged[merged.drug==drug];high=f[(f.dose_nM==10000)&(f.method=='original_loglinear')].copy();high['effect']=abs(high.true_mean_old-high.control_mean_old);candidates=high.nlargest(100,'effect').gene
  scores=[]
  for gene in candidates:
   g=f[(f.gene==gene)&(f.method=='original_loglinear')];rmse_log=np.sqrt(np.mean((g.predicted_mean_old-g.true_mean_old)**2));rmse_ds=np.sqrt(np.mean((g.predicted_mean_dsdr-g.true_mean_dsdr)**2));scores.append((rmse_log-rmse_ds,gene,rmse_log,rmse_ds))
  gain,gene,rlog,rds=max(scores);chosen.append({'drug':drug,'gene':gene,'loglinear_rmse':rlog,'dose_specific_rmse':rds,'rmse_improvement':gain,'selection':'largest DSDR RMSE gain among top100 high-dose DE genes'});g=f[(f.gene==gene)&(f.method=='original_loglinear')].sort_values('dose_nM');hill=f[(f.gene==gene)&(f.method=='improved_hill')].sort_values('dose_nM');ax.plot(g.dose_nM,g.true_mean_old,'o-',color='#222',label='true');ax.plot(g.dose_nM,g.predicted_mean_old,'o-',label='log-linear');ax.plot(hill.dose_nM,hill.predicted_mean_old,'o-',label='Hill');ax.plot(g.dose_nM,g.predicted_mean_dsdr,'o-',label='dose-specific direction');ax.axhline(g.control_mean_old.iloc[0],color='#888',ls='--',label='control');ax.set_xscale('log');ax.set_title(f"{drug.replace(' (GSK1120212)','')}: {gene}");ax.set_xlabel('Dose (nM)');ax.set_ylabel('Mean log-normalized expression');ax.legend(fontsize=8)
 fig.suptitle('Representative genes for the better method');fig.tight_layout();fig.savefig(figdir/'better_method_gene_curves.png',dpi=220,bbox_inches='tight');plt.close(fig);return pd.DataFrame(chosen)


def plot_umap(data,pred,figdir):
 old=sc.read_h5ad(HERE/'prediction_results_task2.h5ad');result=ad.concat([old,pred],join='inner',merge='same',index_unique='-');rng=np.random.default_rng(SEED);chosen=[]
 for _,f in result.obs.groupby(['source','perturbation','dose_value'],observed=True):
  idx=result.obs_names.get_indexer(f.index);chosen.extend(rng.choice(idx,size=min(80,len(idx)),replace=False))
 v=result[np.asarray(chosen)].copy();v.X=v.X.astype(np.float32);sc.tl.pca(v,n_comps=30,svd_solver='arpack',random_state=SEED);sc.pp.neighbors(v,n_neighbors=20,n_pcs=30,random_state=SEED);sc.tl.umap(v,random_state=SEED,min_dist=.35);coords=pd.DataFrame(v.obsm['X_umap'],columns=['UMAP1','UMAP2'],index=v.obs_names);coords=pd.concat([coords,v.obs.reset_index(drop=True).set_index(coords.index)],axis=1);coords.to_csv(HERE/'better_method_umap_coordinates.csv',index_label='cell_id')
 colors={'control':'#aaa','true':'#222','original_loglinear':'#377eb8','improved_hill':'#e41a1c','dose_specific_direction':'#4daf4a'};fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
 for i,drug in enumerate(DRUGS):
  for j,dose in enumerate(DOSES):
   ax=axes[i,j];f=coords[(coords.source=='control')|((coords.perturbation==drug)&(coords.dose_value==dose))]
   for source in colors:
    s=f[f.source==source];ax.scatter(s.UMAP1,s.UMAP2,s=7,alpha=.45,c=colors[source],label=source.replace('_',' '),rasterized=True)
   if i==0:ax.set_title(f'{dose:g} nM')
   if j==0:ax.set_ylabel(drug.replace(' (GSK1120212)','')+'\nUMAP2')
   if i==1:ax.set_xlabel('UMAP1')
   ax.set_xticks([]);ax.set_yticks([])
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.955),ncol=5,frameon=False);fig.suptitle('Three prediction methods in one shared A549 UMAP',y=.995);fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(figdir/'better_method_umap.png',dpi=220,bbox_inches='tight');plt.close(fig)


def mdtable(df):
 h=[str(x) for x in df.columns];lines=['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |'];lines+=['| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None)];return '\n'.join(lines)


def report(comp,genes,runtime):
 top=comp[comp.gene_set=='top100_DE'];summary=[]
 for method in ['improved_hill','dose_specific_direction']:
  r=top[f'r2_gain_{method}_vs_log'];p=top[f'pearson_gain_{method}_vs_log'];summary.append({'method':method,'mean_DE_R2_gain':f'{r.mean():+.4f}','DE_R2_wins_of_8':int((r>1e-10).sum()),'mean_DE_Pearson_gain':f'{p.mean():+.4f}'})
 detail=top[['drug','dose_nM','r2_original_loglinear','r2_improved_hill','r2_dose_specific_direction','r2_gain_dose_specific_direction_vs_log']].copy();
 for c in detail.columns[2:]:detail[c]=detail[c].map(lambda x:f'{x:.4f}')
 gene_text='；'.join(f'{r.drug}: {r.gene}' for r in genes.itertuples())
 text=f"""# 第三种方法：剂量特异方向回归（DSDR）

## 结论

在保留原始 log-linear 和 Hill 方法的基础上，本实验进一步测试 **dose-specific direction regression（DSDR）**。它在 Top 100 DE 基因上的平均 R² 相对原始方法提高 **{top.r2_gain_dose_specific_direction_vs_log.mean():+.4f}**，六个中低剂量条件全部改善，优于 Hill 的平均 **{top.r2_gain_improved_hill_vs_log.mean():+.4f}**。最高提升出现在 Trametinib 1000 nM（{top.r2_gain_dose_specific_direction_vs_log.max():+.4f}）。

{mdtable(pd.DataFrame(summary))}

{mdtable(detail)}

![三方法指标](figures/better_method_metrics.png)

## 方法

原方法假设所有剂量共享最高剂量方向，仅改变标量 $\\alpha(d)$。DSDR 则对每个药物和每个剂量，分别在 K562、MCF7 中计算完整潜在向量 $\\delta_{{c,p,d}}$，再用原 scVIDR 的“对照中心 → 扰动向量”线性回归外推 A549 的 $\\hat\\delta_{{A549,p,d}}$。因此它能表示随剂量改变的方向和基因组合，而不只改变响应强度。拟合过程未使用任何 A549 药物处理细胞。

## 有效范围与代价

DSDR 的优势集中在扰动响应基因：Top 100 DE 的六个中低剂量 R² 全部提升。全部基因的平均 R² 则相对 log-linear变化 {comp[comp.gene_set=='all_retained_genes'].r2_gain_dose_specific_direction_vs_log.mean():+.4f}，说明剂量特异方向增加了对强响应信号的灵活性，但两个源细胞系不足以稳定估计所有背景基因。它不是所有指标全面占优，适合将 DE 基因恢复作为主要目标的场景。

代表基因按 A549 高剂量 Top 100 DE 内、DSDR 相对 log-linear RMSE 改善最大者事后选择（只用于展示）：{gene_text}。

![代表基因](figures/better_method_gene_curves.png)

## 共享 UMAP

对照、真实细胞及三种预测在相同 log-normalized 空间中合并，只拟合一次 PCA/邻接图/UMAP。

![共享 UMAP](figures/better_method_umap.png)

## 文件说明

- `dose_specific_direction.py`：独立 DSDR 实现；原始代码仍在 `original_scvidr_loglinear.py`。
- `run_better_method.py`：端到端复现脚本，复用任务一模型且不覆盖它。
- `metrics_three_methods.csv`、`comparison_three_methods.csv`：三方法原始和配对指标。
- `better_method_predictions.h5ad`、`better_method_per_gene.csv.gz`：DSDR 预测与逐基因均值。
- `better_method_representative_genes.csv`、`better_method_umap_coordinates.csv` 与对应图像：可视化数据。

## 复现

```powershell
conda activate scVIDR
python "任务二/run_better_method.py"
```

运行耗时约 {runtime/60:.1f} 分钟。任务一模型及原始方法文件均保持不变。
""";(HERE/'更好方法.md').write_text(text,encoding='utf-8')


def main():
 started=time.time();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);figdir=HERE/'figures';figdir.mkdir(exist_ok=True);data=sc.read_h5ad(TASK1/'task1_preprocessed.h5ad');model,train=load_preserved_model(data,TASK1/'model');latent=model.get_latent_representation(train);pred=predict(model,train,latent);pred.write_h5ad(HERE/'better_method_predictions.h5ad',compression='gzip');metrics,genes=evaluate(data,pred);genes.to_csv(HERE/'better_method_per_gene.csv.gz',index=False,compression='gzip');old=pd.read_csv(HERE/'metrics.csv');combined=pd.concat([old,metrics],ignore_index=True);combined.to_csv(HERE/'metrics_three_methods.csv',index=False);comp=comparison_table(combined);comp.to_csv(HERE/'comparison_three_methods.csv',index=False);plot_metrics(combined,figdir);oldgenes=pd.read_csv(HERE/'per_gene_mean_expression.csv.gz');chosen=plot_genes(genes,oldgenes,figdir);chosen.to_csv(HERE/'better_method_representative_genes.csv',index=False);plot_umap(data,pred,figdir);runtime=time.time()-started;(HERE/'better_method_run_config.json').write_text(json.dumps({'method':'dose-specific direction regression','seed':SEED,'runtime_seconds':runtime,'task1_model_read_only':True},indent=2),encoding='utf-8');report(comp,chosen,runtime);print(f'Better-method experiment complete in {runtime/60:.1f} min')
if __name__=='__main__':main()
