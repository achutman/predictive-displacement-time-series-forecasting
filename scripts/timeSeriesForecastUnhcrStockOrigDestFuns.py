# -*- coding: utf-8 -*-
"""
Created on Fir 02/04/2021

# Functions used in the main script timeSeriesForecastUnhcrStockOrigDestScript.py

@author: A.Manandhar
Contact: Achut Manandhar, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Migration and Displacement Initiative, Save the Children International

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

def readDatasets(N_LAG,TRAIN_YR_FROM,LABEL_KEY,listFeats):
    print('Reading datasets...')
    ### Datasets common to all countries of origin
    dfs = []
    dfNames = []
    
    # RA stk orig to dest
    if 'RAstk' in LABEL_KEY:
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UNHCR\processed'
        filename = 'unhcr_refugees_asylums_origin_dest_ISO3.csv'
        dfRAstk = pd.read_csv(os.path.join(filepath,filename))
        dfRAstk = dfRAstk.drop(['Unnamed: 0', 'Origin', 'Country / territory of asylum/residence'],axis=1)
        # To allow lagged features
        dfRAstk = dfRAstk.loc[dfRAstk['Year']>=TRAIN_YR_FROM-N_LAG,:]
        dfs.append(dfRAstk)
        dfNames.append('RAstk')
        print('\tRAstk read')
    
    # RA flow orig to dest
    if 'RAflow' in listFeats:
        #dfRAflow = pd.read_excel(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UNHCR\raw\new_ref_arrival_new_asy_app_1962_2019_ISO3.xlsx')
        #dfRAflow = dfRAflow.drop(['Country of asylum','Origin'],axis=1)
        #dfRAflow = dfRAflow.drop([yr for yr in np.arange(1962,TRAIN_YR_FROM)],axis=1)
        ## 1D format
        dfRAflow = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UNHCR\processed\unhcr_refugee_asylum_flow_origin_dest_ISO3_1d.csv')
        dfRAflow = dfRAflow.drop('Unnamed: 0',axis=1)
        dfRAflow = dfRAflow.loc[dfRAflow['Year']>=TRAIN_YR_FROM-N_LAG,:]
        dfs.append(dfRAflow)
        dfNames.append('RAflow')
        print('\tRAflow read')
    
    # R returned from dest to orig
    if 'Rret' in listFeats:
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UNHCR\processed'
        filename = 'unhcr_refugees_returned_origin_dest_ISO3.csv'
        dfRret = pd.read_csv(os.path.join(filepath,filename))
        dfRret = dfRret.drop(['Origin', 'Country / territory of asylum/residence'],axis=1)
        # To allow lagged features
        dfRret = dfRret.loc[dfRret['Year']>=TRAIN_YR_FROM-N_LAG,:]
        dfs.append(dfRret)
        dfNames.append('Rret')
        print('\tRret read')
    
    ## (Host) UCDP Fat
    if 'UcdpFat' in listFeats:
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UCDP\processed'
        filename = 'ucdp-ged-201-deaths-best-Iso3.csv'
        dfUcdpFat = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        dfUcdpFat = dfUcdpFat.drop(['Country'],axis=1)
        dfs.append(dfUcdpFat)
        dfNames.append('UcdpFat')
        print('\tUcdpFat read')
    
    ## (Host) UCDP Evt
    if 'UcdpEvt' in listFeats:
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UCDP\processed'
        filename = 'ucdp-ged-201-event-counts-Iso3.csv'
        dfUcdpEvt = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        dfUcdpEvt = dfUcdpEvt.drop(['Country'],axis=1)
        dfs.append(dfUcdpEvt)
        dfNames.append('UcdpEvt')
        print('\tUcdpEvt read')
        
    ## (Host) Pts
    if 'Pts' in listFeats:
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\Political Terror Scale\processed'
        filename = 'PTS-2019-Iso3-AvgAM.csv'
        dfPts = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        dfPts = dfPts.drop(['Country', 'Country_OLD', 'COW_Code_A', 'COW_Code_N',
                'WordBank_Code_A', 'UN_Code_N', 'Region', 'PTS_A', 'PTS_H', 'PTS_S',
                'NA_Status_A', 'NA_Status_H', 'NA_Status_S'],axis=1)
        dfs.append(dfPts)
        dfNames.append('Pts')
        print('\tPts read')
    
    # Human rights score
    if 'HumRights' in listFeats:
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\Human Rights Score\processed'
        filename = 'data_proc_human_rights_scores.csv'
        #filename = 'data_proc_human_rights_scores_lag1y.csv'
        dfHr = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')    
        dfHr = dfHr.loc[dfHr['Year']>=TRAIN_YR_FROM,:]    
        dfHr = dfHr.drop(['Country'],axis=1)
        dfs.append(dfHr)
        dfNames.append('HumRights')
        print('\tHumRights read')
        
    # (Host) WDI
    # codeKeep = [
    #  'SL_UEM_TOTL_ZS', 
    #  'SP_URB_GROW',
    #  'SP_POP_DPND',
    #  'FP_CPI_TOTL_ZG']
    if ('WdiUem' in listFeats):        
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\World Bank\processed'        
        filename = 'data_proc_WDI_SL_UEM_TOTL_ZS.csv'
        WdiUem = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        WdiUem = WdiUem.drop(['Unnamed: 0','Country'],axis=1)
        WdiUem = WdiUem.loc[WdiUem['Year']>=TRAIN_YR_FROM,:]
        dfs.append(WdiUem)
        dfNames.append('WdiUem')
        print('\tWdiUem read')
        
    if ('WdiUrb' in listFeats):                
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\World Bank\processed'        
        filename = 'data_proc_WDI_SP_URB_GROW.csv'
        WdiUrb = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        WdiUrb = WdiUrb.drop(['Unnamed: 0','Country'],axis=1)
        WdiUrb = WdiUrb.loc[WdiUrb['Year']>=TRAIN_YR_FROM,:]
        dfs.append(WdiUrb)
        dfNames.append('WdiUrb')
        print('\tWdiUrb read')
        
    if ('WdiDep' in listFeats):                
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\World Bank\processed'        
        filename = 'data_proc_WDI_SP_POP_DPND.csv'
        WdiDep = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        WdiDep = WdiDep.drop(['Unnamed: 0','Country'],axis=1)
        WdiDep = WdiDep.loc[WdiDep['Year']>=TRAIN_YR_FROM,:]
        dfs.append(WdiDep)
        dfNames.append('WdiDep')
        print('\tWdiDep read')
        
    if ('WdiCpi' in listFeats):                
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\World Bank\processed'        
        filename = 'data_proc_WDI_FP_CPI_TOTL_ZG.csv'
        WdiCpi = pd.read_csv(os.path.join(filepath,filename),index_col='Iso3')
        WdiCpi = WdiCpi.drop(['Unnamed: 0','Country'],axis=1)
        WdiCpi = WdiCpi.loc[WdiCpi['Year']>=TRAIN_YR_FROM,:]
        dfs.append(WdiCpi)
        dfNames.append('WdiCpi')
        print('\tWdiCpi read')
    
    # Dist
    if ('contig' in listFeats) | ('dist' in listFeats) | ('distw' in listFeats):
        filepath = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\CEPII\processed'
        filename = 'dist_cepii.xls'
        dfDist = pd.read_excel(os.path.join(filepath,filename))
        dfs.append(dfDist)
        dfNames.append('Dist')
        print('\tCEPII dist read')        
        
    # Create a dictionary of dfs
    dfDict = {}
    for i in range(len(dfNames)):
        dfDict[dfNames[i]] = dfs[i]
    
    return dfDict

def mergeOriginOnlyVariables(dfDict,listFeats,orig):
    # Origin only features
    dfsO = []
    dfNamesO = []
    
    # 'AcledUcdpFatMax','AcledUcdpEvtMax'         
    # UCDP Fat
    if 'UcdpFat' in listFeats:
        dfUcdpFat = dfDict['UcdpFat']
        dfUcdpFatO = dfUcdpFat.loc[orig,['Year','Value']].reset_index()
        dfUcdpFatO = dfUcdpFatO.drop('Iso3',axis=1)
        dfUcdpFatO.columns = ['Year','UcdpFat']  
        dfsO.append(dfUcdpFatO)
        dfNamesO.append('UcdpFatO')
        print('\tUcdpFatO generated')
    
    # UCDP Evt
    if 'UcdpEvt' in listFeats:
        dfUcdpEvt = dfDict['UcdpEvt']
        dfUcdpEvtO = dfUcdpEvt.loc[orig,['Year','Value']].reset_index()
        dfUcdpEvtO = dfUcdpEvtO.drop('Iso3',axis=1)
        dfUcdpEvtO.columns = ['Year','UcdpEvt']        
        dfsO.append(dfUcdpEvtO)
        dfNamesO.append('UcdpEvtO')
        print('\tUcdpEvtO generated')
    
    # Pts
    if 'Pts' in listFeats:
        dfPts = dfDict['Pts']
        dfPtsO = dfPts.loc[orig,['Year','Value']].reset_index()
        dfPtsO = dfPtsO.drop('Iso3',axis=1)
        dfPtsO.columns = ['Year','Pts']      
        # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
        dfPtsO = dfPtsO.interpolate(limit=2)
        dfsO.append(dfPtsO)
        dfNamesO.append('PtsO')
        print('\tPtsO generated')
    
    # Human rights index
    if 'HumRights' in listFeats:
        dfHr = dfDict['HumRights']
        dfHrO = dfHr.loc[orig,['Year','Value']].reset_index()
        dfHrO = dfHrO.drop('Iso3',axis=1)    
        dfHrO.columns = ['Year','HumRights'] 
        # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
        dfHrO = dfHrO.interpolate(limit=2)
        dfsO.append(dfHrO)
        dfNamesO.append('dfHrO')
        print('\tdfHrO generated')
            
    # WDIs    
    if 'WdiUem' in listFeats:
        dfWdiUem = dfDict['WdiUem']
        dfWdiUemO = dfWdiUem.loc[orig,['Year','Value']].reset_index()
        dfWdiUemO = dfWdiUemO.drop('Iso3',axis=1)
        perMissing = np.round(100*dfWdiUemO['Value'].isnull().sum()/dfWdiUemO.shape[0])
        print('%d percentage missing'%perMissing)
        dfWdiUemO.columns = ['Year','WdiUem']
        # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
        dfWdiUemO = dfWdiUemO.interpolate(limit=2)
        dfsO.append(dfWdiUemO)
        dfNamesO.append('WdiUemO')
        print('\tWdiUmpO generated')
    
    # WDI    
    if 'WdiUrb' in listFeats:
        dfWdiUrb = dfDict['WdiUrb']
        dfWdiUrbO = dfWdiUrb.loc[orig,['Year','Value']].reset_index()
        dfWdiUrbO = dfWdiUrbO.drop('Iso3',axis=1)
        perMissing = np.round(100*dfWdiUrbO['Value'].isnull().sum()/dfWdiUrbO.shape[0])
        print('%d percentage missing'%perMissing)
        dfWdiUrbO.columns = ['Year','WdiUrb']
        # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
        dfWdiUrbO = dfWdiUrbO.interpolate(limit=2)
        dfsO.append(dfWdiUrbO)
        dfNamesO.append('WdiUrbO')
        print('\tWdiUrbO generated')
    
    # WDI    
    if 'WdiDep' in listFeats:
        dfWdiDep = dfDict['WdiDep']
        dfWdiDepO = dfWdiDep.loc[orig,['Year','Value']].reset_index()
        dfWdiDepO = dfWdiDepO.drop('Iso3',axis=1)
        perMissing = np.round(100*dfWdiDepO['Value'].isnull().sum()/dfWdiDepO.shape[0])
        print('%d percentage missing'%perMissing)
        dfWdiDepO.columns = ['Year','WdiDep']
        # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
        dfWdiDepO = dfWdiDepO.interpolate(limit=2)
        dfsO.append(dfWdiDepO)
        dfNamesO.append('WdiDepO')
        print('\tWdiDepO generated')
    
    # WDI    
    if 'WdiCpi' in listFeats:
        dfWdiCpi = dfDict['WdiCpi']
        dfWdiCpiO = dfWdiCpi.loc[orig,['Year','Value']].reset_index()
        dfWdiCpiO = dfWdiCpiO.drop('Iso3',axis=1)
        perMissing = np.round(100*dfWdiCpiO['Value'].isnull().sum()/dfWdiCpiO.shape[0])
        print('%d percentage missing'%perMissing)
        dfWdiCpiO.columns = ['Year','WdiCpi']
        # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
        dfWdiCpiO = dfWdiCpiO.interpolate(limit=2)
        dfsO.append(dfWdiCpiO)
        dfNamesO.append('WdiCpiO')
        print('\tWdiCpiO generated')
        
    # Start merging
    if len(dfNamesO)==0:
        print('Nothing to merge')
        return np.NaN
    elif len(dfNamesO)==1:
        return dfsO
    else:
        for i in np.arange(1,len(dfNamesO)):
            if i==1:
                dfO = dfsO[0].merge(dfsO[1],left_on='Year',right_on='Year',how='left')    
            else:
                dfO = dfO.merge(dfsO[i],left_on='Year',right_on='Year',how='left')
        return dfO
    
    
def mergeOrigBilateralDestVariables(idest,dest,dfO,dfDict,dfBilateralDict,TRAIN_YR_FROM,TEST_YR_FROM,TOP_N_DEST,listFeats):
    # Merge LABELS and feats corresponding to origin, bilateral, and destination
    dfsOD = []
    dfNamesOD = []
    
    # RA stk orig to dest (as LABELS)      
    dfRAstkO = dfBilateralDict['RAstkO']
    dfRAstkOD = dfRAstkO.loc[dest,['Year','Value']].reset_index()
    # Keep destination name as a column for only this feature
    dfRAstkOD.columns = ['Dest','Year','RAstk']   
    dfsOD.append(dfRAstkOD)
    dfNamesOD.append('RAstkOD')
    print('\tRAstkOD generated')

    # RA flow orig to dest (as features)      
    if 'RAflowO' in dfBilateralDict.keys():
        dfRAflowO = dfBilateralDict['RAflowO']
        if dest in dfRAflowO.index:
            dfRAflowOD = dfRAflowO.loc[dest,['Year','Value']]
            dfRAflowOD.columns = ['Year','RAflow']             
        else:
            print('\tSince RAflowOD n/a, replacing with NaNs')
            dfRAflowOD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'RAflow':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfRAflowOD)
        dfNamesOD.append('RAflowOD')
        print('\tRAflowOD generated')
    
    # Returned ref from dest (host) to origin       
    if 'RretO' in dfBilateralDict.keys():
        dfRretO = dfBilateralDict['RretO']
        if dest in dfRretO.index:
            dfRretOD = dfRretO.loc[dest,['Year','Value']]
            dfRretOD.columns = ['Year','Rret']             
        else:
            print('\tSince RretOD n/a, replacing with NaNs')
            dfRretOD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'Rret':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfRretOD)
        dfNamesOD.append('RretOD')
        print('\tRretOD generated')
    
    # Fat, Evt Host Country
    # 'AcledUcdpFatMaxD', 'AcledUcdpEvtMaxD'
    # UCDP Fat/Evt for host country could be zeros/NaNs
    # UCDP Fat Host Country 
    if 'UcdpFat' in dfDict.keys():
        dfUcdpFat = dfDict['UcdpFat']
        if dest not in dfUcdpFat.index:
            print('\tSince UcdpFatD n/a, replacing with NaNs')
            dfUcdpFatD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'Value':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))}).set_index('Year')  
        elif ((dfUcdpFat.loc[dest,:].shape[0]==2) & (dfUcdpFat.loc[dest,:].shape!=(2,2))):
            dfUcdpFatD = pd.DataFrame({'Year':dfUcdpFat.loc[dest,'Year'],'Value':dfUcdpFat.loc[dest,'Value']},index=[0]).set_index('Year')
        else:
            dfUcdpFatD = dfUcdpFat.loc[dest,:].set_index('Year')        
        dfUcdpFatD = dfUcdpFatD.loc[dfUcdpFatD.index>=TRAIN_YR_FROM,:]
        dfUcdpFatD = dfUcdpFatD.reset_index()
        dfUcdpFatD.columns = ['Year','UcdpFatD']           
        dfsOD.append(dfUcdpFatD)
        dfNamesOD.append('UcdpFatD')
        print('\tUcdpFatD generated')
     
    # UCDP Fat/Evt for host country could be zeros/NaNs
    # UCDP Evt Host Country    
    if 'UcdpEvt' in dfDict.keys():
        dfUcdpEvt = dfDict['UcdpEvt']
        if dest not in dfUcdpEvt.index:
            print('\tSince UcdpEvtD n/a, replacing with NaNs')
            dfUcdpEvtD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'Value':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))}).set_index('Year')
        elif ((dfUcdpEvt.loc[dest,:].shape[0]==2) & (dfUcdpEvt.loc[dest,:].shape!=(2,2))):
            dfUcdpEvtD = pd.DataFrame({'Year':dfUcdpEvt.loc[dest,'Year'],'Value':dfUcdpEvt.loc[dest,'Value']},index=[0]).set_index('Year')
        else:
            dfUcdpEvtD = dfUcdpEvt.loc[dest,:].set_index('Year')
        dfUcdpEvtD = dfUcdpEvtD.loc[dfUcdpEvtD.index>=TRAIN_YR_FROM,:]
        dfUcdpEvtD = dfUcdpEvtD.reset_index()
        dfUcdpEvtD.columns = ['Year','UcdpEvtD'] 
        dfsOD.append(dfUcdpEvtD)
        dfNamesOD.append('UcdpEvtD')
        print('\tUcdpEvtD generated')
    
    # PTS Host Country    
    if 'Pts' in dfDict.keys():
        dfPts = dfDict['Pts']
        if dest in dfPts.index:
            dfPtsD = dfPts.loc[dest,:].set_index('Year')        
            dfPtsD = dfPtsD.loc[dfPtsD.index>=TRAIN_YR_FROM,:]
            dfPtsD = dfPtsD.reset_index()
            dfPtsD.columns = ['Year','PtsD']  
            # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
            print('\tPTS - perform linear interpolation upto 2 years')
            dfPtsD = dfPtsD.interpolate(limit=2)        
        else:
            print('\tSince PtsD n/a, replacing with NaNs')
            dfPtsD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'PtsD':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfPtsD)
        dfNamesOD.append('PtsD')
        print('\tPtsD generated')
    
    
    # Human Rights Host Country
    if 'HumRights' in dfDict.keys():
        dfHr = dfDict['HumRights']
        if dest in dfHr.index:
            dfHrD = dfHr.loc[dest,['Year','Value']]        
            dfHrD.columns = ['Year','HumRightsD'] 
            # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
            print('\tHuman Rights Score - perform linear interpolation upto 2 years')
            dfHrD = dfHrD.interpolate(limit=2)
        else:
            print('\tSince PtsD n/a, replacing with NaNs')
            dfHrD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'HumRightsD':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfHrD)
        dfNamesOD.append('HumRightsD')
        print('\tHumRightsD generated')
                        
    # Wdi Host Country
    if 'WdiUem' in dfDict.keys():
        dfWdiUem = dfDict['WdiUem']
        if dest in dfWdiUem.index:
            dfWdiUemD = dfWdiUem.loc[dest,['Year','Value']]        
            dfWdiUemD.columns = ['Year','WdiUemD'] 
            # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
            print('\tWdiUemD - perform linear interpolation upto 2 years')
            dfWdiUemD = dfWdiUemD.interpolate(limit=2)
        else:
            print('\tSince PtsD n/a, replacing with NaNs')
            dfWdiUemD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'WdiUemD':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfWdiUemD)
        dfNamesOD.append('WdiUemD')
        print('\tWdiUemD generated')
                    
    # Wdi Host Country
    if 'WdiUrb' in dfDict.keys():
        dfWdiUrb = dfDict['WdiUrb']
        if dest in dfWdiUrb.index:
            dfWdiUrbD = dfWdiUrb.loc[dest,['Year','Value']]        
            dfWdiUrbD.columns = ['Year','WdiUrbD'] 
            # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
            print('\tWdiUrbD - perform linear interpolation upto 2 years')
            dfWdiUrbD = dfWdiUrbD.interpolate(limit=2)
        else:
            print('\tSince PtsD n/a, replacing with NaNs')
            dfWdiUrbD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'WdiUrbD':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfWdiUrbD)
        dfNamesOD.append('WdiUrbD')
        print('\tWdiUrbD generated')
        
    # Wdi Host Country
    if 'WdiDep' in dfDict.keys():
        dfWdiDep = dfDict['WdiDep']
        if dest in dfWdiDep.index:
            dfWdiDepD = dfWdiDep.loc[dest,['Year','Value']]        
            dfWdiDepD.columns = ['Year','WdiDepD'] 
            # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
            print('\tWdiDep - perform linear interpolation upto 2 years')
            dfWdiDepD = dfWdiDepD.interpolate(limit=2)
        else:
            print('\tSince PtsD n/a, replacing with NaNs')
            dfWdiDepD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'WdiDepD':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfWdiDepD)
        dfNamesOD.append('WdiDepD')
        print('\tWdiDepD generated')
            
    # Wdi Host Country
    if 'WdiCpi' in dfDict.keys():
        dfWdiCpi = dfDict['WdiCpi']
        if dest in dfWdiCpi.index:
            dfWdiCpiD = dfWdiCpi.loc[dest,['Year','Value']]        
            dfWdiCpiD.columns = ['Year','WdiCpiD'] 
            # Linear interpolate upto 2 years assuming the variable won't change much over 2 years
            print('\tWdiCpi - perform linear interpolation upto 2 years')
            dfWdiCpiD = dfWdiCpiD.interpolate(limit=2)
        else:
            print('\tSince PtsD n/a, replacing with NaNs')
            dfWdiCpiD = pd.DataFrame({'Year':np.arange(TRAIN_YR_FROM,TEST_YR_FROM),'WdiCpiD':np.NaN*np.ones((TEST_YR_FROM-TRAIN_YR_FROM,))})            
        dfsOD.append(dfWdiCpiD)
        dfNamesOD.append('WdiCpiD')
        print('\tWdiCpiD generated')
                            
    # Dist
    if 'DistO' in dfBilateralDict.keys():        
        dfDistO = dfBilateralDict['DistO']
        if dest in dfDistO['iso_d'].values:
            dfDistOD = dfDistO.loc[(dfDistO['iso_d']==dest),:]
        else:
            print('\tSince DistOD n/a, replacing with NaNs')
            dfDistOD = pd.DataFrame({'iso_d':dest},index=[0])
            for key in dfDistO.keys():
                if key!='iso_d':
                    dfDistOD[key] = np.NaN
        print('\tDistOD generated')

    # Start merging
    # There should be at least one dataframe (LABELS) to merge
    if len(dfNamesOD)==0:
        print('Error - There should be at least one dataframe (LABELS) to merge!')
        return dfO
    else:
        dfOD = dfO.merge(dfsOD[0],left_on='Year',right_on='Year',how='left')    
    if len(dfNamesOD)>1:
        for i in np.arange(1,len(dfNamesOD)):
            dfOD = dfOD.merge(dfsOD[i],left_on='Year',right_on='Year',how='left')                
    
    # Replace NaN Dest names
    if 'Dest' not in dfOD.keys():
        print('Error - Dest key should already be in the merged data frame!')
    else:
        dfOD['Dest'] = dfOD['Dest'].fillna(dest)
    
    if 'DistO' in dfBilateralDict.keys():
        dfOD = dfOD.merge(dfDistOD,left_on='Dest',right_on='iso_d',how='left').drop('iso_d',axis=1)        
    
    # Destination country = a feature
    if 'DestId' in listFeats:
        dfOD['DestId'] = (TOP_N_DEST-idest)*np.ones(dfOD.shape[0],)      
        
    return dfOD
    
def findTopNDestCountriesFun(dfRAstkO,topNDest):
    totalRefAsyPerDest = dfRAstkO.loc[:,['ISO3_D','Value']].groupby('ISO3_D').sum().sort_values('Value',ascending=False)    
    topNDestNames = totalRefAsyPerDest.index.values[:topNDest]
    dfRAstkO = dfRAstkO.set_index('ISO3_D')
    return dfRAstkO.loc[topNDestNames,:],topNDestNames

def prepare_data_fun(df,Nahead,labelKey,Nlag,lagFeats):
#def prepare_data_fun(df,Nwin,Nahead,labelKey):
#Convert into supervised format, i.e. [X,Y]
#Nahead = 3 # how many years ahead to forecast
#Nwin = 3 # how many years of historical data to use

    # print(Nlag,lagFeats)
    # Concatenate Nwin-lagged features
    if lagFeats!=[]:    
        for key in lagFeats:
            for nwin in np.arange(Nlag):            
                colName = '%sLag%dy'%(key,nwin+1)       
                df[colName] = df[key].shift(nwin+1)
        # Drop nan rows                
        df = df.drop(np.arange(Nlag))
    # Concatenate Nahead LABELS    
    for nAhd in np.arange(1,Nahead+1):    
        colName = '%sLead%dy'%(labelKey,nAhd)
        df[colName] = df['%s'%labelKey].shift(-nAhd)    
    
    # Lets not drop rows with NaN for now!!
    
#    # Drop past Nwin-1 years because there will be NaNs    
#    idxDrop = dfCntyXY.index[np.arange(Nwin-1)]
#    # Drop future Nahead-1 years because there will be NaNs
#    idxDrop = idxDrop.append(dfCntyXY.index[-Nahead:])
#    print(dfCntyXY.shape)
#    dfCntyXY = dfCntyXY.drop(idxDrop)
#    print(dfCntyXY.shape)      
#    dfCntyXY = dfCntyXY.sort_index() # sort by year
    return df  

def remFeaturesWithTooManyMissingValues(dfCatXY,dictFeats,listFeats,FEATS_REM_PER_MISS):
    # Remove features with more than x% missing values
    # Except the autoregressive feature, i.e. don't remove RAstk
    featsRem = []
    Nsamples = dfCatXY.shape[0]
    print('Percentage missing')
    for feat in listFeats:
        perMissing = np.round_(100*dfCatXY[feat].isnull().sum()/Nsamples)
        print('\t%s, %d'%(feat,perMissing))
        if ((feat!='RAstk') & (perMissing>FEATS_REM_PER_MISS)):
            featsRem.append(feat)
    if len(featsRem)>0:
        print('Removing one or more features...')
        for feat in featsRem:
            # Remove from dictFeats
            for key in dictFeats.keys():
                if feat in dictFeats[key]:
                    print('\tRemove %s from %s'%(feat,key))
                    dictFeats[key].remove(feat)
            # Remove from dfCatXY
            dfCatXY = dfCatXY.drop(feat,axis=1)
        # Update listFeats
        listFeats = []
        for key in dictFeats.keys():
            if dictFeats[key]!=[]:
                listFeats.extend(dictFeats[key])        
    return dfCatXY,dictFeats,listFeats

def generateListColTrans(dictFeats,topNDestNames):
    listFeats = []
    listTrans = []
    for key in dictFeats.keys():    
        if dictFeats[key]!=[]:              
            if key=='feats1Hot':
                listTrans.append(('f_1hot', OneHotEncoder(dtype='int'), dictFeats[key]))
                listFeats.extend(topNDestNames)
            if key=='featsZmuv':
                listTrans.append(('f_zmuv', StandardScaler(), dictFeats[key]))
                listFeats.extend(dictFeats[key])
            if key=='featsQt':
                Nquantiles = 10
                qt_out_dist = 'normal'
                listTrans.append(('f_quant', QuantileTransformer(n_quantiles=Nquantiles,output_distribution=qt_out_dist), dictFeats[key]))
                listFeats.extend(dictFeats[key])
            if key=='featsPt':
                listTrans.append(('f_pt', PowerTransformer(method='yeo-johnson', standardize=False), dictFeats[key]))
                listFeats.extend(dictFeats[key])
            if key=='featsPass':
                listTrans.append(('f_pass', 'passthrough', dictFeats[key]))
                listFeats.extend(dictFeats[key])
    column_trans = ColumnTransformer(listTrans)
    return column_trans,listFeats

def quantileTransformerFun(train, test, Nquantiles, qt_out_dist, plotOption):
    # Plot before
    if plotOption:
        for nDim in np.arange(train.shape[1]):
            plt.hist(train[:,nDim],density=True,alpha=.5)
            plt.hist(test[:,nDim],density=True,alpha=.5)
            plt.title('Before Dim %d'%nDim)
            plt.show()
    
    qt = QuantileTransformer(n_quantiles=Nquantiles,output_distribution=qt_out_dist)
    qt = qt.fit(train)
    train = qt.fit_transform(train)
    test = qt.fit_transform(test)
    
    # Plot after
    if plotOption:
        for nDim in np.arange(train.shape[1]):
            plt.hist(train[:,nDim],density=True,alpha=.5)
            plt.hist(test[:,nDim],density=True,alpha=.5)
            plt.title('After Dim %d'%nDim)
            plt.show()
    return train, test, qt

   
def save_forecasts_fun(pathSaveFull,dfCatXY,Yout,LABELS):
# Save model output, i.e. the forecasts (Yout) N-y ahead per spreadsheet
# Saving model outputs highly reccommended
# Will be super useful to re-generate plots/scores/etc.
#
#dfCatXY = supervised data, i.e [features,LABELS] or [X,Y]
#idtest_x = indices corresponding to test data
#test_y = true test values, i.e. true RefFlow values
#Yout = estimated test values, i.e. estimtaed RefFlow values, maybe available for multiple iterations
    Nahead = len(LABELS)    
    # Save N-yr ahead Your per sheet
    with pd.ExcelWriter(pathSaveFull) as writer:
        for nAhd in np.arange(Nahead):      
            if Nahead==1:
                Yout_nahead = pd.DataFrame(Yout.T)    
            else:
                Yout_nahead = pd.DataFrame(Yout[:,:,nAhd].T) 
            # Yout_nahead = pd.DataFrame(Yout[:,:,nAhd].T) 
            # Compute mean, std, 95% confidence interval of estimates over multiple iterations
            Yout_nahead['Mean'] = Yout_nahead.mean(axis=1)
            Yout_nahead['Std'] = Yout_nahead.std(axis=1)
            Yout_nahead['MMinus'] = Yout_nahead['Mean']-1.96*Yout_nahead['Std']
            Yout_nahead['MPlus'] = Yout_nahead['Mean']+1.96*Yout_nahead['Std']        
            Yout_nahead['Truth'] = dfCatXY.loc[dfCatXY['Test'],LABELS[nAhd]].values
            Yout_nahead['Orig'] = dfCatXY.loc[dfCatXY['Test'],'Orig'].values
            Yout_nahead['Dest'] = dfCatXY.loc[dfCatXY['Test'],'Dest'].values
            # Yout_nahead['DestId'] = dfCatXY.loc[dfCatXY['Test'],'DestId'].values
            # Remember to update "years" to represent lead years that we are predicting!
            Yout_nahead['Year'] = dfCatXY.loc[dfCatXY['Test'],'Year'].values + (nAhd+1)
            # Update year index N-year ahead, e.g.
            # In 2000, 1-yr ahead will forecast for 2001,
            # In 2000, 2-yr ahead will forecast for 2002,
            # In 2000, 3-yr ahead will forecast for 2003
            dfCatXY.to_excel(writer,sheet_name='Data')
            Yout_nahead.to_excel(writer,sheet_name='%dyAhead'%(nAhd+1))                
    print('Saved')
    return True

def plot_per_nahead_out(dfCatXY,Yout_nahead,nAhd,LABELS,savefig,pathSave,folderSave):
# Plot per N-Year ahead forecast
# Save plots (optional but highly reccommended)    
#
#dfCatXY = supervised data, i.e [features,LABELS] or [X,Y]
#Yout_nahead = estimated and other statistics over multiple iterations generated by save_forecasts()    
#nAhd = n-year ahead, e.g. 1
#savefig = True or False
#pathSave/folderSave = path to save figures
    
    for cnty in dfCatXY['Dest'].unique():                
        # plot true training RefAsyStk data            
        # Shift dfCatXY year by nAhd years forward
        idxTrain = ((dfCatXY['Dest']==cnty) & dfCatXY['Train'])
        plt.plot(dfCatXY.loc[idxTrain,'Year'].values+(nAhd+1),dfCatXY.loc[idxTrain,LABELS[nAhd]],'o-')
        # plot true testing RefAsyFlow data
        idxTest = ((dfCatXY['Dest']==cnty) & dfCatXY['Test'])
        plt.plot(dfCatXY.loc[idxTest,'Year'].values+(nAhd+1),dfCatXY.loc[idxTest,LABELS[nAhd]],'o-')
        # plot estimated testing RefAsyStk data
        # YoutCnty year index already shifted  
        YoutCnty = Yout_nahead.loc[Yout_nahead['Dest']==cnty,:].set_index('Year')                
        plt.plot(YoutCnty['Mean'],'o-k')
        plt.fill_between(YoutCnty.index.values, YoutCnty['MMinus'], YoutCnty['MPlus'], color='k', alpha=.20)                     
        plt.xlim([1988,2022])
        plt.title('%s'%cnty)
        plt.ylabel('Ref & Asy Stock')        
        plt.legend(['Train','Truth','Estimated Mean','95%CI'])
        if savefig:
            # Save each figure using filename = dest country name            
            pathSaveFull = os.path.join(pathSave,folderSave,'%dyAhead'%(nAhd+1),'%s'%cnty)
            # pathSaveFull = os.path.join(pathSave,folderSave,'%dyAheadOpt'%(nAhd+1),'%s'%cnty)
            print(pathSaveFull)  
            plt.savefig(pathSaveFull,dpi=100)       
        plt.show()
               
def plot_per_nahead_out_cluster(dfCatXY,Yout_nahead,nAhd,LABELS,savefig,pathSave):
# Plot per N-Year ahead forecast
# Save plots (optional but highly reccommended)    
#
#dfCatXY = supervised data, i.e [features,LABELS] or [X,Y]
#Yout_nahead = estimated and other statistics over multiple iterations generated by save_forecasts()    
#nAhd = n-year ahead, e.g. 1
#savefig = True or False
#pathSave/folderSave = path to save figures
    
    for orig in dfCatXY['Orig'].unique():                
        for dest in dfCatXY.loc[dfCatXY['Orig']==orig,'Dest'].unique():                
            # plot true training RefAsyStk data            
            # Shift dfCatXY year by nAhd years forward
            idxTrain = ((dfCatXY['Orig']==orig) & (dfCatXY['Dest']==dest) & dfCatXY['Train'])
            plt.plot(dfCatXY.loc[idxTrain,'Year'].values+(nAhd+1),dfCatXY.loc[idxTrain,LABELS[nAhd]],'o-')
            # plot true testing RefAsyFlow data
            idxTest = ((dfCatXY['Orig']==orig) & (dfCatXY['Dest']==dest) & dfCatXY['Test'])
            plt.plot(dfCatXY.loc[idxTest,'Year'].values+(nAhd+1),dfCatXY.loc[idxTest,LABELS[nAhd]],'o-')
            # plot estimated testing RefAsyStk data
            # YoutCnty year index already shifted  
            YoutCnty = Yout_nahead.loc[(Yout_nahead['Orig']==orig) & (Yout_nahead['Dest']==dest),:].set_index('Year')                
            plt.plot(YoutCnty['Mean'],'o-k')
            plt.fill_between(YoutCnty.index.values, YoutCnty['MMinus'], YoutCnty['MPlus'], color='k', alpha=.20)                     
            plt.xlim([1988,2022])
            plt.title('%s'%dest)
            plt.ylabel('Ref & Asy Stock')        
            plt.legend(['Train','Truth','Estimated Mean','95%CI'])
            if savefig:
                # Save each figure using filename = dest country name            
                pathSaveFull = os.path.join(pathSave,orig,'%dyAhead'%(nAhd+1),'%s'%dest)                
                print(pathSaveFull)  
                plt.savefig(pathSaveFull,dpi=100)       
            plt.show()        

# def plot_per_nahead_out_all_dest(dfCatXY,Yout_nahead,nAhd,LABELS,savefig,pathSave,folderSave):
# # Plot per N-Year ahead forecast
# # Save plots (optional but highly reccommended)    
# #
# #dfCatXY = supervised data, i.e [features,LABELS] or [X,Y]
# #Yout_nahead = estimated and other statistics over multiple iterations generated by save_forecasts()    
# #nAhd = n-year ahead, e.g. 1
# #savefig = True or False
# #pathSave/folderSave = path to save figures
    
#     for cnty in dfCatXY['Dest'].unique():                
#         # plot true RefAsyStk data            
#         # Shift dfCatXY year by nAhd years forward
#         idxTruth = (dfCatXY['Dest']==cnty)
#         plt.plot(dfCatXY.loc[idxTruth,'Year'].values+(nAhd+1),dfCatXY.loc[idxTruth,LABELS[nAhd]],'C0')        
#         # plot estimated testing RefAsyStk data
#         # YoutCnty year index already shifted  
#         YoutCnty = Yout_nahead.loc[Yout_nahead['Dest']==cnty,:].set_index('Year')                
#         plt.plot(YoutCnty['Mean'],'xr')
#         plt.fill_between(YoutCnty.index.values, YoutCnty['MMinus'], YoutCnty['MPlus'], color='r', alpha=.20)                     
#         plt.xlim([1988,2022])
# #    plt.title('%s'%cnty)
# #    plt.ylabel('Ref & Asy Stock')        
# #    plt.legend(['Train','Truth','Estimated Mean','95%CI'])
# #    if savefig:
# #        # Save each figure using filename = dest country name            
# #        pathSaveFull = os.path.join(pathSave,folderSave,'%dyAhead'%(nAhd+1),'%s'%cnty)
# #        print(pathSaveFull)  
# #        plt.savefig(pathSaveFull,dpi=100)       
#     plt.show()