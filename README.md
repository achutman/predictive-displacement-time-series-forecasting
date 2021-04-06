# predictive-displacement-time-series-forecasting
•	Learns a regression model to predict the stock of refugees and asylum seekers at a host country from a country of origin
•	Can be modified to predict the flow of refugees and asylum seekers from origin to host country

scripts
•	timeSeriesForecastUnhcrStockOrigDestScript.py learns a model per country of origin
•	timeSeriesForecastUnhcrStockClusterOrigDestScript.py learns a model per cluster of countries of origin
•	Uses functions defined in timeSeriesForecastUnhcrStockOrigDestFuns.py
•	Scoring script in timeSeriesForecastUnhcrStockOrigDestScoreScript.py

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


# Methodolodgy/Algorithm/Pseudocode
timeSeriesForecastUnhcrStockClusterOrigDestScript.py
(1) Libraries, configuration settings, model parameters
(1.1) Import necessary libraries
(1.2) Define all parameters and choose various settings 
(2) Data load, prepare, process
(2.1) Read all datasets
(2.2) Learn a model per cluster of countries of origin (clustering code in github)
(2.3) For each cluster, loop over selected countries
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
(2.12) (Ignore this step, infeasible to consider every orig-dest pair as a feature... Drop destination country that is not in training set?
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
