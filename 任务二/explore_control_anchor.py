#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test target-control-anchored decoder calibration without treated-target leakage."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from run_task2 import CONTROL,DOSES,DRUGS,TARGET,decode,load_preserved_model,seed_everything,target_maximum_delta
from original_scvidr_loglinear import predict_latent_loglinear
import scanpy as sc

def main():
 seed_everything(); data=sc.read_h5ad(ROOT/'任务一'/'task1_preprocessed.h5ad'); model,train=load_preserved_model(data,ROOT/'任务一'/'model'); latent=model.get_latent_representation(train)
 tm=train.obs; mask=(tm.cell_line.to_numpy()==TARGET)&(tm.perturbation.to_numpy()==CONTROL); zc=latent[mask]
 observed=np.asarray(train.X[mask].mean(axis=0)).reshape(-1); reconstructed=decode(model,zc).mean(axis=0); correction=observed-reconstructed
 pd.DataFrame({'gene':train.var_names,'observed_control_mean':observed,'reconstructed_control_mean':reconstructed,'correction':correction}).to_csv(HERE/'control_anchor_corrections.csv.gz',index=False,compression='gzip')
 rows=[]
 for drug in DRUGS:
  ctrl,delta=target_maximum_delta(latent,train.obs,drug); preds=predict_latent_loglinear(ctrl,delta,DOSES)
  for dose,z in preds.items():
   pm=(decode(model,z)+correction).mean(axis=0); truth=data[(data.obs.cell_line==TARGET)&(data.obs.perturbation==drug)&(data.obs.dose_value==dose)]; true=np.asarray(truth.X.mean(axis=0)).reshape(-1); de=np.zeros(data.n_vars,bool);de[np.argsort(-abs(true-observed))[:100]]=1;h=data.var.hvg_rank.to_numpy()<=500
   for gs,m in {'all_retained_genes':np.ones(data.n_vars,bool),'top500_HVG':h,'top100_DE':de}.items(): rows.append({'drug':drug,'dose_nM':dose,'gene_set':gs,'pearson_anchor':stats.pearsonr(true[m],pm[m])[0],'r2_anchor':r2_score(true[m],pm[m])})
 r=pd.DataFrame(rows);b=pd.read_csv(HERE/'metrics_comparison.csv');r=b.merge(r,on=['drug','dose_nM','gene_set']);r['r2_gain_vs_loglinear']=r.r2_anchor-r.r2_original_loglinear;r['pearson_gain_vs_loglinear']=r.pearson_anchor-r.pearson_original_loglinear;r.to_csv(HERE/'control_anchor_exploration_metrics.csv',index=False)
 print('mean abs correction',abs(correction).mean());print(r.groupby('gene_set')[['r2_gain_vs_loglinear','pearson_gain_vs_loglinear']].agg(['mean',lambda x:(x>0).sum()]));print(r[r.gene_set=='all_retained_genes'][['drug','dose_nM','r2_gain_vs_loglinear','pearson_gain_vs_loglinear']].to_string(index=False))
if __name__=='__main__':main()
