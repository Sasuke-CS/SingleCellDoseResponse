#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a fixed training-DE-gated log-linear/Hill hybrid."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUNTIME=HERE/".runtime_cache"; RUNTIME.mkdir(exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR",str(RUNTIME)); os.environ.setdefault("MPLCONFIGDIR",str(RUNTIME))
import scanpy as sc
DRUGS=["(+)-JQ1","Trametinib (GSK1120212)"]
DOSES=[10.,100.,1000.,10000.]


def main():
    data=sc.read_h5ad(ROOT/"任务一"/"task1_preprocessed.h5ad")
    train=data[data.obs["split"]=="train"]
    means=pd.read_csv(HERE/"per_gene_mean_expression.csv.gz")
    rows=[]; gate_rows=[]
    for drug in DRUGS:
        # Drug-level gate: mean absolute response across both training lines and
        # all doses; sign agreement downweights cell-line-specific effects.
        effects=[]
        for line in ["K562","MCF7"]:
            ctrl=np.asarray(train[(train.obs.cell_line==line)&(train.obs.perturbation=="control")].X.mean(axis=0)).reshape(-1)
            line_effect=[]
            for dose in DOSES:
                tr=np.asarray(train[(train.obs.cell_line==line)&(train.obs.perturbation==drug)&(train.obs.dose_value==dose)].X.mean(axis=0)).reshape(-1)
                line_effect.append(tr-ctrl)
            effects.append(np.mean(line_effect,axis=0))
        e1,e2=effects
        score=(np.abs(e1)+np.abs(e2))/2 * np.where(np.sign(e1)==np.sign(e2),1.0,0.25)
        rank=np.empty(len(score),int); rank[np.argsort(-score)]=np.arange(1,len(score)+1)
        # Fixed top quartile gate: no A549 response is consulted.
        weight=np.where(rank<=500,1.0,0.0)
        for gene,r,s,w in zip(data.var_names,rank,score,weight):
            gate_rows.append({"drug":drug,"gene":gene,"training_de_rank":r,"training_de_score":s,"hill_weight":w})
        for dose in DOSES:
            a=means[(means.drug==drug)&(means.dose_nM==dose)]
            orig=a[a.method=="original_loglinear"].set_index("gene")
            hill=a[a.method=="improved_hill"].set_index("gene")
            pm=orig.predicted_mean.to_numpy()*(1-weight)+hill.predicted_mean.to_numpy()*weight
            tm=orig.true_mean.to_numpy(); cm=orig.control_mean.to_numpy()
            de=np.zeros(len(tm),bool);de[np.argsort(-np.abs(tm-cm))[:100]]=True
            hvg=orig.hvg_rank.to_numpy()<=500
            for gene_set,mask in {"all_retained_genes":np.ones(len(tm),bool),"top500_HVG":hvg,"top100_DE":de}.items():
                rows.append({"drug":drug,"dose_nM":dose,"gene_set":gene_set,
                    "pearson_hybrid":stats.pearsonr(tm[mask],pm[mask])[0],"r2_hybrid":r2_score(tm[mask],pm[mask])})
    result=pd.DataFrame(rows)
    baseline=pd.read_csv(HERE/"metrics_comparison.csv")
    result=baseline.merge(result,on=["drug","dose_nM","gene_set"])
    result["pearson_gain_vs_loglinear"]=result.pearson_hybrid-result.pearson_original_loglinear
    result["r2_gain_vs_loglinear"]=result.r2_hybrid-result.r2_original_loglinear
    result.to_csv(HERE/"de_gated_hybrid_exploration_metrics.csv",index=False)
    pd.DataFrame(gate_rows).to_csv(HERE/"de_gated_gene_weights.csv.gz",index=False,compression="gzip")
    print(result.groupby('gene_set')[["r2_gain_vs_loglinear","pearson_gain_vs_loglinear"]].agg(['mean',lambda x:(x>0).sum()]))

if __name__=="__main__":main()
