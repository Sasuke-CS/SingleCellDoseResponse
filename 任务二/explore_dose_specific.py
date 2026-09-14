#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate dose-specific latent-direction regression on the task-2 test set."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE))
from run_task2 import CONTROL,DOSES,DRUGS,TARGET,decode,load_preserved_model,mean_rows,seed_everything
import scanpy as sc

def main():
 seed_everything();data=sc.read_h5ad(ROOT/'任务一'/'task1_preprocessed.h5ad');model,train=load_preserved_model(data,ROOT/'任务一'/'model');latent=model.get_latent_representation(train);o=train.obs
 target_ctrl=(o.cell_line.to_numpy()==TARGET)&(o.perturbation.to_numpy()==CONTROL);zt=latent[target_ctrl];rows=[]
 for drug in DRUGS:
  for dose in DOSES:
   cs=[];ds=[]
   for line in ['K562','MCF7']:
    cm=(o.cell_line.to_numpy()==line)&(o.perturbation.to_numpy()==CONTROL);dm=(o.cell_line.to_numpy()==line)&(o.perturbation.to_numpy()==drug)&(o.dose_value.to_numpy()==dose);c=mean_rows(latent[cm]);cs.append(c);ds.append(mean_rows(latent[dm])-c)
   delta=LinearRegression().fit(cs,ds).predict([mean_rows(zt)])[0];pm=decode(model,zt+delta).mean(axis=0);truth=data[(data.obs.cell_line==TARGET)&(data.obs.perturbation==drug)&(data.obs.dose_value==dose)];tm=np.asarray(truth.X.mean(axis=0)).reshape(-1);ctrl=np.asarray(data[(data.obs.cell_line==TARGET)&(data.obs.perturbation==CONTROL)].X.mean(axis=0)).reshape(-1);de=np.zeros(data.n_vars,bool);de[np.argsort(-abs(tm-ctrl))[:100]]=1;h=data.var.hvg_rank.to_numpy()<=500
   for gs,m in {'all_retained_genes':np.ones(data.n_vars,bool),'top500_HVG':h,'top100_DE':de}.items():rows.append({'drug':drug,'dose_nM':dose,'gene_set':gs,'pearson_dose_specific':stats.pearsonr(tm[m],pm[m])[0],'r2_dose_specific':r2_score(tm[m],pm[m])})
 r=pd.DataFrame(rows);b=pd.read_csv(HERE/'metrics_comparison.csv');r=b.merge(r,on=['drug','dose_nM','gene_set']);r['r2_gain_vs_loglinear']=r.r2_dose_specific-r.r2_original_loglinear;r['pearson_gain_vs_loglinear']=r.pearson_dose_specific-r.pearson_original_loglinear;r.to_csv(HERE/'dose_specific_exploration_metrics.csv',index=False);print(r.groupby('gene_set')[['r2_gain_vs_loglinear','pearson_gain_vs_loglinear']].agg(['mean',lambda x:(x>0).sum()]));print(r[r.gene_set=='top100_DE'][['drug','dose_nM','r2_gain_vs_loglinear']].to_string(index=False))
if __name__=='__main__':main()
