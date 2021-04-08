# -*- coding: utf-8 -*-
"""
Created on Fir 02/04/2021

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

This work was done in collaboration with University of Virginia
Contact:
Michele Claibourn, DIRECTOR OF SOCIAL, NATURAL, ENGINEERING AND DATA SCIENCES, UVA LIBRARY
David Leblang, PROFESSOR OF POLITICS AND PUBLIC POLICY, DIRECTOR OF THE GLOBAL POLICY CENTER, Frank Batten School of Leadership and Public Policy

# Model per country of origin
# Main script to load, prepare, process data and train/test model
# Learns a regression model to predict the stock of refugees and asylum seekers at a host country from a country of origin
# Can be modified to predict the flow of refugees and asylum seekers from origin to host country
# Uses functions defined in timeSeriesForecastUnhcrStockOrigDestFuns.py
# Scoring script in timeSeriesForecastUnhcrStockOrigDestScoreScript.py

# Methodolodgy/Algorithm/Pseudocode
timeSeriesForecastUnhcrStockOrigDestScript.py
(1) Libraries, configuration settings, model parameters
(1.1) Import necessary libraries
(1.2) Define all parameters and choose various settings
(2) Data load, prepare, process
(2.1) Read all datasets
(2.2) Select countries to model, either manually or based on INFORM risk index
(2.3) Loop over selected countries
(2.4) Generate features specific to the country of origin, 
      e.g. UCDP fatalities at origin, Human Rights Score at origin
(2.5) Generate Origin to Destination bilateral labels and features   
      Note destination = host country
      e.g. Stock of refugees and asylum seekers at host from the origin,
      Stock of returned refugees from host to the country of origin,
(2.6) For each country of origin, loop over top N host countries to merge 
      labels and all features - origin-specific, bilateral, and destination-specific
(2.7) Merge labels and all features - origin-specific, bilateral, and destination-specific
(2.8) Include lag features, lead labels
Feature processing
(2.9) Drop features with too many missing values determined by PER_FEAT_MISSING
(2.10) If not imputing missing (NaN) feature values, drop samples with any missing feature values    
(2.11) Divide dataset into training and testing set
       Currently, no validation set, but include if want to optimize model parameters
(2.12) Drop destination country that is not in training set?
(2.13) Different transformation for different features
(2.14) If imputing, impute missing values now
(2.15) Still may have NaN values in training "labels"
       Drop training samples with missing labels
(3) Model training and testing
(3.1) Loop over multiple iterations of model training/testing
(3.2) Choose a linear/non-linear model or appropriate pipeline
(3.3) ***Key model training step***
(3.4) ***Key prediction step***
(4) Save and plot outputs
(4.1) Save data and forecasts      
(4.2) Plot feature importance
(4.3) Plot per N-Year ahead forecast
(4.4) Print summary statistics

# Relevant documentations
https://scikit-learn.org/stable/modules/preprocessing.html
https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_0_24_0.html
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
https://www.statsmodels.org/stable/mixed_linear.html
https://stats.idre.ucla.edu/other/mult-pkg/introduction-to-linear-mixed-models/
https://github.com/statsmodels/statsmodels/wiki/Examples#linear-mixed-models


@author: A.Manandhar
Contact: Achut Manandhar, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Migration and Displacement Initiative, Save the Children International
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import xlrd

# (1) Import necessary libraries
from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectFromModel
# from scipy.stats import randint
# from sklearn.model_selection import RandomizedSearchCV

# from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge
# from sklearn.linear_model import PoissonRegressor
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.neural_network import MLPRegressor
# from sklearn.svm import LinearSVR
#from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr

from sklearn.impute import KNNImputer

# cd "C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\codes\tutorialTimeSeriesForecasting"
from timeSeriesForecastUnhcrStockOrigDestFuns import *
    

############################################################################### 
# (2) Define all parameters and choose various settings
TOP_N_DEST = 10
# Lag years
N_LAG = 0
# Features for which lag years to consider
LAG_FEATS = []
# LAG_FEATS = ['RAstk']
# N years ahead forecast
N_AHEAD = 1
# Training years from
TRAIN_YR_FROM = 1989
# Testing years from
TEST_YR_FROM = 2015
# Min training samples required to proceed
MIN_TRAIN_SAMPLES = 100
# Min testing samples required to proceed
MIN_TEST_SAMPLES = 10
# Remove features with more than x% missing values
FEATS_REM_PER_MISS = 10
# Data imputation option
DATA_IMPUTATION = True
# Data imputation based on KNN imputer parameter (could use other types of imputer)
KNN_IMPUTER_K = 5
# labels
#LABEL_KEY = 'RAflow'
LABEL_KEY = 'RAstk'
#LABELS = ['RAflowLead1y','RAflowLead2y','RAflowLead3y']    
LABELS = ['RAstkLead1y']    
# LABELS = ['RAstkLead1y','RAstkLead2y','RAstkLead3y']    
# features
# Supports different transformation methods for different features
# Default setting does not use any feature transformation
# OneHot
feats1Hot = []
# feats1Hot = ['Dest']
# Standard scaler
featsZmuv = []
#featsZmuv = ['distw']
# featsZmuv = ['RAstk','RAflow','Rret']
# Minmax
# featsMinMax = []
# featsMinMax = ['dist']
# featsMinMax = ['distw']
# Quantile transformation
# Nquantiles = 10
# qt_out_dist = 'uniform'
# qt_out_dist = 'normal'
featsQt = []
#featsQt = ['RAflow']
#featsQt = ['RAflow','AcledUcdpFatMax', 'AcledUcdpEvtMax']
# featsQt = ['RAstk']
# featsQt = ['RAstk','RAstkLag1y']
# featsQt = ['RAstk','AcledUcdpFatMax', 'AcledUcdpEvtMax']
# featsQt = ['RAstk','AcledUcdpFatMax', 'AcledUcdpEvtMax','dist','distw']
# Power transformation
featsPt = []
#featsPt = ['AcledUcdpFatMax','AcledUcdpEvtMax','AcledUcdpFatMaxD','AcledUcdpEvtMaxD']
# Features to use without any transformation
# featsPass = []
#featsPass = ['RAstk','DestId']
featsPass = ['RAstk','UcdpFat', 'UcdpEvt', 'Pts', 'HumRights', 
             'WdiUem', 'WdiUrb', 'WdiDep', 'WdiCpi', 
             'RAflow', 'Rret', 'UcdpFatD', 'UcdpEvtD', 'PtsD', 'HumRightsD', 
             'contig', 'dist', 'WdiUemD', 'WdiUrbD', 'WdiDepD', 'WdiCpiD', 'DestId']
# Dictionary of subsets of features by feature transformation type
dictFeats = {'feats1Hot':feats1Hot,'featsZmuv':featsZmuv,'featsQt':featsQt,'featsPt':featsPt,'featsPass':featsPass}
# List of features to include in the model
listFeats = []
for key in dictFeats.keys():
    if dictFeats[key]!=[]:
        listFeats.extend(dictFeats[key])
# Number of iterations of model training and testing
N_ITERS = 30
# Save / plot option
SAVE_OUTPUTS = False


###############################################################################
# (3) Read all datasets
dfDict = readDatasets(N_LAG,TRAIN_YR_FROM,LABEL_KEY,listFeats)


###############################################################################
# (4) Select countries to model, either manually or based on INFORM risk index
# Select countries, loop over countries
# SSD, SDN, SYR, BGD    
# Loop over origin countries
# countries = ['BDI','ETH','SOM','SSD','CAF','TCD','COG','COD','LBY','MLI','NGA','SDN','IRQ','SYR','YEM','AFG','BGD','IRN','MMR','VEN']    
# countries = ['BDI','ETH','SOM','CAF','TCD','COG','COD','LBY','MLI','NGA','SDN','IRQ','YEM','AFG','BGD','IRN','MMR','VEN'] # 'SSD', 'SYR'   
# countries = ['AFG', 'CAF', 'CIV', 'COD', 'SOM', 'TCD', 'UGA', 'YEM',
#              'BDI', 'CMR', 'ETH', 'KEN', 'MLI', 'NER', 'NGA', 'SDN',
#              'BGD', 'GTM', 'HND', 'HTI', 'IND', 'MMR', 'PAK', 'PHL',
#              'BOL', 'CRI', 'CUB', 'DOM', 'ECU', 'JAM', 'NIC','PAN', 'PER', 'SLV',
# countries = ['COL', 'GEO', 'ISR', 'JOR', 'LBN', 'TUR']
# Select countries based on INFORM risk index
pathInform = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM'
dfInform = pd.read_csv(os.path.join(pathInform,'raw','INFORM2020_TREND_2010_2019_v040_ALL_2 INFORMRiskIndex.csv'))
countries = dfInform.loc[dfInform['2020']>=5,'Iso3'].values
cntySkip = ['PRK','PNG','PSE']
for cnty in countries:
    if cnty in cntySkip:              
        countries = np.delete(countries,np.where(countries==cnty))

# Define path to save all model outputs and plots
# pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\StkLagODtop10'
#pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\StkLagDestODtop10'
pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\time series forecast\UNHCR\RAStkOrgDst\StkLagDestFlowRetFatEvtPtsHrDistWdi1_4ODtop10'

# Skip certain destination countries for which majority of datasets are n/a  
# destSkip = ['SSD','HKG','CUW']
destSkip = ['HKG','CUW']

###############################################################################
# Generate summary statistics 
dfSummStat = pd.DataFrame({'Country':countries,
                           'NtopDest':np.NaN*np.ones(len(countries),),
                           'Nfeatures':np.NaN*np.ones(len(countries),),
                           'Ntrain':np.NaN*np.ones(len(countries),),
                           'Ntest':np.NaN*np.ones(len(countries),),
                           'Nimputed':np.NaN*np.ones(len(countries),),
                           'Ndropped':np.NaN*np.ones(len(countries)),
                           'Modelled':np.zeros(len(countries))}).set_index('Country')

# (5) Loop over selected countries
for orig in countries:    
    print('\n%s'%orig)
    ###########################################################################
    # (6) Generate features specific to the country of origin
    # E.g. UCDP fatalities at origin, Human Rights Score at origin
    # dfO is a dictionary of data frames
    dfO = mergeOriginOnlyVariables(dfDict,listFeats,orig)
    print(orig)
    print(dfO.isnull().sum())
    
    ###########################################################################    
    # (7) Generate Origin to Destination bilateral labels and features   
    # Note destination = host country
    # E.g. Stock of refugees and asylum seekers at host from the origin,
    # Stock of returned refugees from host to the country of origin,
    # Remittance flow from host to the country of origin (currently not 
    # included because data only available from 2010 onwards)
    
    # Stock of refugees and asylum seekers at host from the origin
    dfRAstkO = dfDict['RAstk'].loc[(dfDict['RAstk']['ISO3_O']==orig),['ISO3_D','Year','Value']]
    # Top N destination countries
    dfRAstkO,topNDestNames = findTopNDestCountriesFun(dfRAstkO,TOP_N_DEST)          
    for dest in topNDestNames:
        if dest in destSkip:
            print('Removing destination country due to data unavailability')            
            topNDestNames = np.delete(topNDestNames,np.where(topNDestNames==dest))
            
    # Flow of refugees and asylum seekers from the origin to the host
    dfRAflowO = dfDict['RAflow'].loc[(dfDict['RAflow']['ISO3_O']==orig),:].drop('ISO3_O',axis=1).set_index('ISO3_D')
    
    # Stock of returned refugees from host to the country of origin,
    dfRretO = dfDict['Rret'].loc[(dfDict['Rret']['ISO3_O']==orig),['ISO3_D','Year','Value']].set_index('ISO3_D')    
    
    # CEPII distance between origin and host countries
    dfDistO = dfDict['Dist'].loc[(dfDict['Dist']['iso_o']==orig),:].drop('iso_o',axis=1).set_index('iso_d')    
    featsDist = [key for key in dfDistO.keys() if key in listFeats]    
    dfDistO = dfDistO.loc[:,featsDist]
    dfDistO = dfDistO.reset_index()
    
    # dfBilateralDict is a dictionary of data frames
    dfBilateralDict = {'RAstkO':dfRAstkO,'RAflowO':dfRAflowO,'RretO':dfRretO,'DistO':dfDistO}
    
    # (8) For each country of origin, loop over top N host countries to merge 
    # labels and all features - origin-specific, bilateral, and destination-specific
    dfCatXY = pd.DataFrame()
    for idest, dest in enumerate(topNDestNames):    
        print('\n')
        print(orig,dest)        
        
        # (9) Merge labels and all features - origin-specific, bilateral, and destination-specific
        dfOD = mergeOrigBilateralDestVariables(idest,dest,dfO,dfDict,dfBilateralDict,TRAIN_YR_FROM,TEST_YR_FROM,TOP_N_DEST,listFeats)                
        print(dfOD.isnull().sum())                        
                
        # (10) Include lag features, lead labels
        # Include lag features? 
        # Include after determining appropirate lag years per feature, e.g. by performing cross-correlation        
        # Include lead labels, i.e. N-year ahead label to predict
        dfOD = prepare_data_fun(dfOD,N_AHEAD,LABEL_KEY,N_LAG,LAG_FEATS)
    
        # Append over top N host countries
        dfCatXY = dfCatXY.append(dfOD)                     
        
    # Add Origin column
    dfCatXY['Orig'] = [orig for i in np.arange(dfCatXY.shape[0])]    
    # Print a summary of label-feature data frame and NaN values
    print(dfCatXY.shape)
    print(dfCatXY.isnull().sum())
        
    ###########################################################################
    # (11) Feature processing
    # How to take care of missing data???
    # The more the features, the more missing data!!!
    
    # Ad-hoc - assume UCDP NaN = 0
    print('\tUCDP Fat and Evt - Assuming NaN = 0')
    for key in ['UcdpFat','UcdpEvt','UcdpFatD','UcdpEvtD']:
        dfCatXY[key] = dfCatXY[key].fillna(0.)
                
    # Uncomment to plot and debug
    # # Feat / label distribution
    # for key in dfCatXY.keys():
    #     if dfCatXY[key].isnull().sum()>0:
    #         print('Removing %d NaN instances'%dfCatXY[key].isnull().sum())
    #         plt.hist(dfCatXY.loc[np.logical_not(dfCatXY[key].isnull()),key]),plt.title(key),plt.show()
    #     else:
    #         plt.hist(dfCatXY[key]),plt.title(key),plt.show()
    # # Plot cross-correlation
    # for key in dfCatXY.keys():
    #     if key not in ['Year','RAstk','Dest','DestId','contig','RAstkLead1y','Train','Test']:                        
    #         plt.xcorr(dfCatXY.loc[dfCatXY['Train'],'RAstk'].values, dfCatXY.loc[dfCatXY['Train'],key].values, usevlines=True, maxlags=10, normed=True, lw=2)
    #         plt.grid(True)
    #         plt.title('Xcorr[RAstk,%s]'%key)
    #         plt.show()

    # (12) Drop features with too many missing values determined by PER_FEAT_MISSING
    print(dfCatXY.shape)
    dfCatXY, dictFeats, listFeats = remFeaturesWithTooManyMissingValues(dfCatXY,dictFeats,listFeats,FEATS_REM_PER_MISS)    
    print(dfCatXY.shape)
    print(dfCatXY.isnull().sum())
    
    # (13) Either impute missing (NaN) feature values of drop samples with any NaN feature values    
    if DATA_IMPUTATION:
        # If imputing missing feature values, imputing will be done later
        print('Keeping NaN values because will perform data imputation later')
    else:
        # Drop samples whose any feature has a NaN value
        # Summary stat
        dfSummStat.loc[orig,'Ndropped'] = (dfCatXY.loc[:,listFeats].isnull().sum(axis=1)>0).sum()
        print('Removing all samples (rows) with any NaN value features')
        print(dfCatXY.shape)
        dfCatXY = dfCatXY.loc[(dfCatXY.loc[:,listFeats].isnull().sum(axis=1)==0),:]
        print(dfCatXY.shape)    
        # Reset index
        dfCatXY = dfCatXY.reset_index()   
        dfCatXY = dfCatXY.drop('index',axis=1)  
    
    ###########################################################################                
    # (14) Divide dataset into training and testing set
    # Currently, no validation set, but include if want to optimize model parameters
    # Train/Test
    dfCatXY['Train'] = (dfCatXY['Year']<TEST_YR_FROM)
    dfCatXY['Test'] = (dfCatXY['Year']>=TEST_YR_FROM)
    
    # Only proceed if there are MIN_TRAIN_SAMPLES of training samples and MIN_TEST_SAMPLES of test samples
    if (dfCatXY['Train'].sum()<MIN_TRAIN_SAMPLES):
        print('Skipping %s - Training samples < %s'%(orig,MIN_TRAIN_SAMPLES))
        continue
    elif (dfCatXY['Test'].sum()<MIN_TEST_SAMPLES):
        print('Skipping %s - Test samples < %s'%(orig,MIN_TEST_SAMPLES))
        continue
    else:    
        # Proceed         
        # (15) Drop destination country that is not in training set?    
        cntyTrain = dfCatXY.loc[dfCatXY['Train'],'Dest'].unique()    
        cntyRem = [cnty for cnty in topNDestNames if cnty not in cntyTrain]
        if len(cntyRem)>0:
            for cnty in cntyRem:
                topNDestNames = np.delete(topNDestNames,np.where(topNDestNames==cnty))
            
        # (16) Different tranformation for different features
        column_trans,listFeats = generateListColTrans(dictFeats,topNDestNames)            
        # # Debug - plot transformed features
        # train_x = column_trans.fit_transform(dfCatXY.loc[dfCatXY['Train'],:])
        # for i in np.arange(train_x.shape[1]):
        #     plt.hist(train_x[:,i]),plt.title(listFeats[i]),plt.show()
        
        # (17) If imputing, impute missing values now
        if DATA_IMPUTATION:        
            # Summary stat
            dfSummStat.loc[orig,'Nimputed'] = dfCatXY.loc[:,listFeats].isnull().sum().sum()
            print('Imputing missing values...')
            imputer = KNNImputer(n_neighbors=KNN_IMPUTER_K)
            imputer = imputer.fit(dfCatXY.loc[dfCatXY['Train'],listFeats],dfCatXY.loc[dfCatXY['Train'],LABELS])
            print(dfCatXY.loc[dfCatXY['Train'],listFeats].isnull().sum())
            dfCatXY.loc[dfCatXY['Train'],listFeats] = imputer.transform(dfCatXY.loc[dfCatXY['Train'],listFeats])
            print(dfCatXY.loc[dfCatXY['Train'],listFeats].isnull().sum())
            print(dfCatXY.loc[dfCatXY['Test'],listFeats].isnull().sum())
            dfCatXY.loc[dfCatXY['Test'],listFeats] = imputer.transform(dfCatXY.loc[dfCatXY['Test'],listFeats])
            print(dfCatXY.loc[dfCatXY['Test'],listFeats].isnull().sum())        
            
        # (18) Still may have NaN values in training "labels"
        # Drop traning samples with missing labels
        # It is ok to have test samples with missing labels
        # Summary stat
        if np.isnan(dfSummStat.loc[orig,'Ndropped']):
            dfSummStat.loc[orig,'Ndropped'] = ((dfCatXY['Year']<TEST_YR_FROM)&dfCatXY['RAstkLead1y'].isnull()).sum()
        else:
            dfSummStat.loc[orig,'Ndropped'] = dfSummStat.loc[orig,'Ndropped']+ ((dfCatXY['Year']<TEST_YR_FROM)&dfCatXY['RAstkLead1y'].isnull()).sum()
        print(dfCatXY.isnull().sum())
        dfCatXY = dfCatXY.loc[np.logical_not((dfCatXY['Year']<TEST_YR_FROM)&dfCatXY['RAstkLead1y'].isnull()),:]
        print(dfCatXY.isnull().sum())
        
        # Summary stats
        dfSummStat.loc[orig,'NtopDest'] = len(dfCatXY['Dest'].unique())
        dfSummStat.loc[orig,'Nfeatures'] = len(listFeats)
        dfSummStat.loc[orig,'Ntrain'] = dfCatXY['Train'].sum()
        dfSummStat.loc[orig,'Ntest'] = dfCatXY['Test'].sum()
        
        # Only proceed if there are MIN_TRAIN_SAMPLES of training samples and MIN_TEST_SAMPLES of test samples
        if (dfCatXY['Train'].sum()<MIN_TRAIN_SAMPLES):
            print('Skipping %s - Training samples < %s'%(orig,MIN_TRAIN_SAMPLES))
            continue
        elif (dfCatXY['Test'].sum()<MIN_TEST_SAMPLES):
            print('Skipping %s - Test samples < %s'%(orig,MIN_TEST_SAMPLES))
            continue
        else:    
            # Proceed 
            # Summary stat
            dfSummStat.loc[orig,'Modelled'] = True
            # # If performing label transformation      
            # train_y = dfCatXY.loc[dfCatXY['Train'],LABELS].values  
            # train_x = dfCatXY.loc[dfCatXY['Train'],listFeats].values
            # test_y = dfCatXY.loc[dfCatXY['Test'],LABELS].values      
            # test_x = dfCatXY.loc[dfCatXY['Test'],listFeats].values
            #
            # Label transformation (power)
            # pt_label = PowerTransformer(method='yeo-johnson', standardize=False)
            # pt_label = pt_label.fit(train_y)
            # train_y = pt_label.transform(train_y)
            # test_y = pt_label.transform(test_y)
            # plt.hist(train_y)
            # plt.hist(test_y)
            # # Label transformation (quantiles)    
            # train_y, test_y, qtLABELS = quantileTransformerFun(train_y, test_y, Nquantiles, qt_out_dist, True)
              
            # # Linear mixed effects model
            # from statsmodels.regression.mixed_linear_model import MixedLM    
            # # idx_exog = [idx for idx in np.arange(len(features)) if features[idx] in ['RAstk','AcledUcdpFatMax', 'AcledUcdpEvtMax', 'PTS', 'PTSD', 'HumRights', 'HumRightsD']]  
            # idx_exog = [idx for idx in np.arange(len(features)) if features[idx] in ['RAstk','AcledUcdpFatMax', 'AcledUcdpEvtMax', 'PTS', 'HumRights']]  
            # idx_exog_re = [idx for idx in np.arange(len(features)) if features[idx] in ['PTSD', 'HumRightsD']]  
            # idx_groups = [idx for idx in np.arange(len(features)) if features[idx] in ['Dest']]  
            # features = np.array(features)    
            # print(features[idx_exog])    
            # print(features[idx_exog_re])    
            # print(features[idx_groups])    
            # train_endog = train_y[:,0]
            # train_exog = train_x[:,idx_exog].astype(np.float)    
            # train_exog_re = train_x[:,idx_exog_re].astype(np.float)    
            # train_groups = dfCatXY.loc[dfCatXY['Train'],'DestId'].values    
            # # md1 = MixedLM(train_endog,train_exog,train_groups)
            # md1 = MixedLM(train_endog,train_exog,train_groups,train_exog_re)
            # mdf1 = md1.fit()
            # print(mdf1.summary())    
                
            # (19) Loop over multiple iterations of model training/testing
            Yout = []
            for niter in np.arange(N_ITERS):        
                # (20) Choose a linear/non-linear model or appropriate pipeline
                # model = RandomForestRegressor(n_estimators=100)    
                # model = GradientBoostingRegressor()
                # Feature selection + regression
                # model = Pipeline([('feature_selection', SelectFromModel(GradientBoostingRegressor())),  
                #                   ('regression', RandomForestRegressor(n_estimators=100))])             
                # Feature processing + regression
                # model = Pipeline([('feature_proc', column_trans),  
                #                    ('regression', Ridge())])
                # model = Pipeline([('feature_proc', column_trans),  
                #                     ('regression', LinearRegression())])
                model = Pipeline([('feature_proc', column_trans),  
                                    ('regression', RandomForestRegressor(n_estimators=100))])
                # model = Pipeline([('feature_proc', column_trans),  
                #                     ('regression', GradientBoostingRegressor(n_estimators=100))])
                # model = Pipeline([('feature_impute', KNNImputer(n_neighbors=KNN_IMPUTER_K)),
                                    # ('feature_proc', column_trans),  
                                    # ('regression', RandomForestRegressor(n_estimators=100))])
                # (21) ***Key model training step***
                model = model.fit(dfCatXY.loc[dfCatXY['Train'],listFeats],dfCatXY.loc[dfCatXY['Train'],LABELS])  
                # ***The trained model***
                model_fit = model.named_steps['regression']
                
                # # Example script to pptmize Random Forest regressor        
                # param_dist = {"max_depth": [3, None],
                #       "max_features": randint(1, train_x.shape[1]+1),
                #       "min_samples_split": randint(2, train_x.shape[1]+1)}
                # # Optmize Gradient Boosting        
                # param_dist = {"max_depth": [2, 10],
                #       "max_features": randint(1, train_x.shape[1]+1),
                #       "min_samples_split": randint(2, train_x.shape[1]+1)}
                # model = RandomizedSearchCV(estimator=GradientBoostingRegressor(),param_distributions=param_dist)
                # model = model.fit(train_x,train_y[:,0])
                # model_fit = model.best_estimator_
                # print(model_fit)
                
                # Debug - plot gradient boosting Feat importance
                importances = model_fit.feature_importances_        
                # listFeats = np.concatenate((np.array(['StkLag','Fat','Evt','Pts','PtsD','Hr','HrD']),topNDestNames))        
                plt.figure(figsize=(7,4))
                plt.title(("%s - Feature importances"%orig),fontsize ='xx-large')
                plt.bar(range(len(importances)), importances)
                plt.xticks(range(len(importances)), labels=listFeats,rotation=90)  
                plt.show()
                    
                # (22) ***Key prediction step***
                forecasts = model.predict(dfCatXY.loc[dfCatXY['Test'],listFeats])
                # Inverse transform
                # forecasts = pt_label.inverse_transform(forecasts.reshape(len(forecasts),1))
                # forecasts = qtLABELS.inverse_transform(forecasts)
                # Debug- plot in original space
                for nAhd in np.arange(N_AHEAD):
                    # If multi-output, will need to index dfCatXY.loc[dfCatXY['Train'],LABELS] and forecasts appropriately
                    plt.plot(dfCatXY.loc[dfCatXY['Test'],LABELS].values,'x')
                    plt.plot(forecasts,'.-')
                    plt.legend(['Truth','Estimate'])
                    plt.title('%d-Year Ahead'%(nAhd+1))
                    plt.show()   
                Yout.append(forecasts)
            Yout = np.array(Yout)
                            
            if SAVE_OUTPUTS:
                # (23) Save data and forecasts      
                # Make a directory per country to save model outputs and plots    
                os.mkdir(os.path.join(pathSave,orig))
                for nAhd in np.arange(N_AHEAD):
                    os.mkdir(os.path.join(pathSave,orig,'%dyAhead'%(nAhd+1)))   
                folderSave = orig
                # fileSave = 'YoutRegRidgeStkOrgDst%dyAhead.xlsx'%(N_AHEAD)
                fileSave = 'YoutRegRAStkOrgDst%dyAhead.xlsx'%(N_AHEAD)
                # fileSave = 'YoutRegRAStkOrgDst%dyAheadOpt.xlsx'%(N_AHEAD)
                pathSaveFull = os.path.join(pathSave,folderSave,fileSave)
                print(pathSaveFull)
                flag = save_forecasts_fun(pathSaveFull,dfCatXY,Yout,LABELS)
                
                # (24) Plot feature importance
                # Plot LR coef
                # plt.figure(figsize=(7,4))
                # plt.title("Feature coefficients",fontsize ='xx-large')
                # plt.bar(np.arange(len(listFeats)),model_fit.coef_.flatten())
                # plt.xticks(range(len(listFeats)), labels=listFeats, rotation=90)    
                # plt.savefig(os.path.join(pathSave,folderSave,'RidgeCoef.png'),dpi=100)
                # Plot random forest Feat importance
                importances = model_fit.feature_importances_
                imp_std = np.std([tree.feature_importances_ for tree in model_fit.estimators_],axis=0)             
                # listFeats = ['StkLag','Host']    
                # listFeats = np.concatenate((np.array(featsQt),np.array(featsMinMax),np.array(featsPass)))
                plt.figure(figsize=(7,4))
                plt.title("Feature importances",fontsize ='xx-large')
                plt.bar(range(len(imp_std)), importances,
                        color="C0", yerr=imp_std, align="center")
                plt.xticks(range(len(imp_std)), labels=listFeats, rotation=90)    
                plt.savefig(os.path.join(pathSave,folderSave,'RFfeatsImp.png'),dpi=100)
                # plt.savefig(os.path.join(pathSave,folderSave,'RFfeatsImpOpt.png'),dpi=100)
                plt.show()
                
                # # from sklearn.inspection import permutation_importance
                # # result = permutation_importance(model_fit, test_x, test_y[:,0], n_repeats=10)
                # # sorted_idx = result.importances_mean.argsort()
                # # plt.subplot(1, 2, 2)
                # # plt.boxplot(result.importances[sorted_idx].T,vert=False, LABELS=features[sorted_idx])
                # # plt.title("Permutation Importance (test set)")
                # # fig.tight_layout()
                # # plt.show()
            
                   
                ###############################################################################
                # (25) Plot per N-Year ahead forecast
                # Save plots (optional but highly reccommended)
                pathSaveFull = os.path.join(pathSave,folderSave,fileSave)
                print(pathSaveFull)                
                for nAhd in [0]:#np.arange(N_AHEAD):      
                    Yout_nahead = pd.read_excel(pathSaveFull,sheet_name='%dyAhead'%(nAhd+1))        
                    savefig=True   
                    plot_per_nahead_out(dfCatXY,Yout_nahead,nAhd,LABELS,savefig,pathSave,folderSave)
                
                # ##############################################################################
                # # Currently unverified
                # # Score (RMSE, CC) per N-Year ahead forecast
                # # Save scores (optional but highly reccommended)
                # for nAhd in np.arange(Nahead):  
                #     Yout_nahead = pd.read_excel(os.path.join(pathSave,folderSave,fileSave),sheet_name='%dyAhead'%(nAhd+1))            
                #     dfScoreRmse, dfScoreCc = score_per_nahead_out(Yout_nahead,N_ITERSModelLearning)
                #     with pd.ExcelWriter(os.path.join(pathSave,folderSave,'%dyAhead'%(nAhd+1),'scoreRmseCc.xlsx')) as writer:
                #         dfScoreRmse.to_excel(writer,sheet_name='rmse')
                #         dfScoreCc.to_excel(writer,sheet_name='cc')
    
# Print summary statistics 
print(dfSummStat)
# dfSummStat.to_csv(os.path.join(pathSave,'dataGenSummaryStatisitcs.csv'))
