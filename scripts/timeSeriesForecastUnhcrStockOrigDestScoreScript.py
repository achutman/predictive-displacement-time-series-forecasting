# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:27:05 2020

@author: A.Manandhar
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from scipy.stats import pearsonr

def aggCorrCoefFun(grp):
    grpNoNan = grp.dropna(axis=0)  
    if grpNoNan.shape[0]<2:
        return np.NaN
    else:
        return pearsonr(grpNoNan['Mean'].values,grpNoNan['Truth'].values)[0]

# Define settings and parameters to score
N_AHEAD = 1
MISS_FORECAST_THRESH = 40000

###############################################################################
# Score Orig Dest modelled per country of origin
pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\StkLagDestFlowRetFatEvtPtsHrDistWdi1_4ODtop10'
countries = os.listdir(pathSave)
# with pd.ExcelWriter(os.path.join(pathSave,'score.xlsx')) as writer:    
    for nAhd in np.arange(N_AHEAD):      
        Yscore = pd.DataFrame()
        for orig in countries:
            folderSave = orig        
            fileSave = 'YoutRegRAStkOrgDst%dyAhead.xlsx'%(N_AHEAD)
            pathSaveFull = os.path.join(pathSave,folderSave,fileSave)
            print(pathSaveFull)               
            Yout_N_AHEAD = pd.read_excel(pathSaveFull,sheet_name='%dyAhead'%(nAhd+1))
            Yout_N_AHEAD['MAE']=np.round_(np.abs(Yout_N_AHEAD['Truth'].values-Yout_N_AHEAD['Mean'].values),1)
            Yout_N_AHEAD['MAPE']=np.round_(100*np.abs(Yout_N_AHEAD['Truth'].values-Yout_N_AHEAD['Mean'].values)/Yout_N_AHEAD['Truth'].values,1)
            Yout_N_AHEAD['MissedForecasts'] = (Yout_N_AHEAD['MAE']>MISS_FORECAST_THRESH).astype(np.float)
            YoutAgg = Yout_N_AHEAD.loc[:,['Dest','MAE','MAPE']].groupby('Dest').mean()        
            YoutAgg['MissedForecasts'] = Yout_N_AHEAD.loc[:,['Dest','MissedForecasts']].groupby('Dest').sum()        
            grouped = Yout_N_AHEAD.loc[:,['Dest','Mean','Truth']].groupby('Dest')
            YoutAgg['CorrCoef'] = grouped.apply(aggCorrCoefFun).values
            YoutAgg['Origin'] = [orig for i in np.arange(YoutAgg.shape[0])]        
            YoutAgg = YoutAgg.reset_index()
            Yscore = Yscore.append(YoutAgg)
        Yscore = Yscore.reset_index()
        Yscore = Yscore.drop('index',axis=1)
        Yscore = Yscore.set_index('Origin')
        # Yscore.to_excel(writer,sheet_name='%dyAhead'%(nAhd+1))

###############################################################################
# Score Orig Dest modelled per cluster of countries of origin
pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\ClusterStkLagDestFlowRetFatEvtPtsHrDistWdi1_4ODtop10'
clustersConsidered = [0,1,2,5,9,11,16]
for nCluster in clustersConsidered:
    folderSave = 'Cluster%d'%nCluster
    pathSaveFull = os.path.join(pathSave,folderSave,'YoutRegRAStkOrgDst1yAhead.xlsx')
    print(pathSaveFull)  
    with pd.ExcelWriter(os.path.join(pathSave,folderSave,'score.xlsx')) as writer:    
        for nAhd in np.arange(N_AHEAD):
            Yout_N_AHEAD = pd.read_excel(pathSaveFull,sheet_name='%dyAhead'%(nAhd+1))  
            Yout_N_AHEAD['MAE']=np.round_(np.abs(Yout_N_AHEAD['Truth'].values-Yout_N_AHEAD['Mean'].values),0)
            Yout_N_AHEAD['MAPE']=np.round_(100*np.abs(Yout_N_AHEAD['Truth'].values-Yout_N_AHEAD['Mean'].values)/Yout_N_AHEAD['Truth'].values,0)
            Yout_N_AHEAD['MissedForecasts'] = (Yout_N_AHEAD['MAE']>MISS_FORECAST_THRESH).astype(np.float)
            YoutAgg = Yout_N_AHEAD.loc[:,['Orig','Dest','MAE','MAPE']].groupby(['Orig','Dest']).mean()        
            YoutAgg['MissedForecasts'] = Yout_N_AHEAD.loc[:,['Orig','Dest','MissedForecasts']].groupby(['Orig','Dest']).sum()        
            grouped = Yout_N_AHEAD.loc[:,['Orig','Dest','Mean','Truth']].groupby(['Orig','Dest'])
            YoutAgg['CorrCoef'] = grouped.apply(aggCorrCoefFun).values    
            YoutAgg = YoutAgg.reset_index()
            YoutAgg = YoutAgg.set_index('Orig')
            YoutAgg.to_excel(writer,sheet_name='%dyAhead'%(nAhd+1))

# Merge all countries of origin over clusters
pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\ClusterStkLagDestFlowRetFatEvtPtsHrDistWdi1_4ODtop10'
clustersConsidered = [0,1,2,5,9,11,16]
Yout_merged = pd.DataFrame()
for nCluster in clustersConsidered:
    folderSave = 'Cluster%d'%nCluster
    pathSaveFull = os.path.join(pathSave,folderSave,'score.xlsx')
    print(pathSaveFull)  
    Yout_score = pd.read_excel(pathSaveFull)  
    Yout_merged = Yout_merged.append(Yout_score)
Yout_merged = Yout_merged.set_index('Orig')
# Yout_merged.to_excel(os.path.join(pathSave,'score.xlsx'),sheet_name='1yAhead')

###############################################################################
# Compare scores - Modelled per country of origin vs. Modelled per cluster of countries of origin
pathSave0 = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\StkLagDestFlowRetFatEvtPtsHrDistWdi1_4ODtop10'
pathSave1 = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\ClusterStkLagDestFlowRetFatEvtPtsHrDistWdi1_4ODtop10'  
nAhd = 0
Yscore0 = pd.read_excel(os.path.join(pathSave0,'score.xlsx'),sheet_name='%dyAhead'%(nAhd+1))
Yscore1 = pd.read_excel(os.path.join(pathSave1,'score.xlsx'),sheet_name='%dyAhead'%(nAhd+1))
print(Yscore0.describe())
print(Yscore1.describe())

# Keep common countries only
countriesCommon = np.intersect1d(Yscore0['Origin'].unique(),Yscore1['Orig'].unique())
Yscore0 = Yscore0.set_index('Origin')
Yscore0 = Yscore0.loc[countriesCommon,:]
Yscore0 = Yscore0.reset_index()
Yscore0 = Yscore0.sort_values(['Origin','Dest'])
Yscore1 = Yscore1.set_index('Orig')
Yscore1 = Yscore1.loc[countriesCommon,:]
Yscore1 = Yscore1.reset_index()
Yscore1 = Yscore1.sort_values(['Orig','Dest'])
print(Yscore0.shape)
print(Yscore1.shape)

# Comparison of MAPE
MAPEthresh = 100
fig,ax=plt.subplots(figsize=(6,5))
ax.plot(Yscore0['MAPE'],Yscore1['MAPE'],'o')
ax.plot([0,MAPEthresh],[0,MAPEthresh],':k')
ax.set_xlim([-1,MAPEthresh])
ax.set_ylim([-1,MAPEthresh])
ax.set_xlabel(r'$\%$ Error (Model per Country of Origin)')
ax.set_ylabel(r'$\%$ Error (Model per Cluster of Countries of Origin)')
ax.set_title('Below diagonal line means clustering is better')
# fig.savefig(os.path.join(pathSave1,'scoreComparisonMAPE.png'),dpi=300)

# Comparison of Corr Coeff.
CCthresh = 1
fig,ax=plt.subplots(figsize=(6,5))
ax.plot(Yscore0['CorrCoef'],Yscore1['CorrCoef'],'o')
ax.plot([-CCthresh,CCthresh],[-CCthresh,CCthresh],':k')
ax.set_xlim([-CCthresh,CCthresh])
ax.set_ylim([-CCthresh,CCthresh])
ax.set_xlabel('Corr Coef (Model per Country of Origin)')
ax.set_ylabel('Corr Coef (Model per Cluster of Countries of Origin)')
ax.set_title('Above diagonal line means clustering is better')
# fig.savefig(os.path.join(pathSave1,'scoreComparisonCC.png'),dpi=300)

# # Other ways to compare???
# plt.plot(Yscore1['MAE']<Yscore0['MAE'],'.')
# print('CC Cases better/worse than baseline')
# print((Yscore1['CorrCoef']>Yscore0['CorrCoef']).sum())
# print((Yscore1['CorrCoef']<Yscore0['CorrCoef']).sum())
# print('MAE Cases better/worse than baseline')
# print((Yscore1['MAE']<Yscore0['MAE']).sum())
# print((Yscore1['MAE']>Yscore0['MAE']).sum())
# print('Total MAE')
# print(Yscore0['MAE'].sum())
# print(Yscore1['MAE'].sum())
# print('Missed Forecasts')
# print(Yscore0['MissedForecasts'].sum())
# print(Yscore1['MissedForecasts'].sum())

# plt.hist(Yscore0['CorrCoef'],alpha=.25)
# plt.hist(Yscore1['CorrCoef'],alpha=.25)
# plt.show()
# plt.hist(Yscore0['MAE'],alpha=.25)
# plt.hist(Yscore1['MAE'],alpha=.25)
# plt.show()
# plt.hist(Yscore1['CorrCoef']-Yscore0['CorrCoef'])
# plt.show()
# plt.hist(Yscore1['MAE']-Yscore0['MAE'])
# plt.show()

# plt.plot(Yscore0['CorrCoef'],'x')
# plt.plot(Yscore1['CorrCoef'])
# plt.plot(Yscore0['MAE'],'x')
# plt.plot(Yscore1['MAE'])

# plt.plot(Yscore0['MAE'],'.')
# plt.plot(Yscore1['MAE'],'.')
# # plt.plot(Yscore2['MAE'],'.')
# plt.ylim([0,50000])

# plt.hist(np.log10(Yscore0.loc[Yscore0['MAE']!=0,'MAE']),alpha=.25)
# plt.hist(np.log10(Yscore1.loc[Yscore1['MAE']!=0,'MAE']),alpha=.25)
# plt.hist(np.log10(Yscore2['MAE']),alpha=.25)

# plt.plot(Yscore0['MAE'],Yscore0['CorrCoef'],'.')
# plt.plot(Yscore1['MAE'],Yscore1['CorrCoef'],'.')
# plt.xlim([0,50000])
