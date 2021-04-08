# predictive-displacement-time-series-forecasting
- Learns a regression model to predict the stock of refugees and asylum seekers at a host country from a country of origin
- Can be modified to predict the flow of refugees and asylum seekers from origin to host country

The project is inspired from the following works:
- https://github.com/IBM/mixed-migration-forecasting
- https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/

scripts
- timeSeriesForecastUnhcrStockOrigDestScript.py learns a model per country of origin
- timeSeriesForecastUnhcrStockClusterOrigDestScript.py learns a model per cluster of countries of origin
- Uses functions defined in timeSeriesForecastUnhcrStockOrigDestFuns.py
- Scoring script in timeSeriesForecastUnhcrStockOrigDestScoreScript.py

outputs
- Provides a selection of example outputs

data
- All data used in this work are publicly available from corresponding original sources
- Processed data will be made available via HDX https://data.humdata.org/, whose link will be updated here soon.

Methodolodgy/Algorithm/Pseudocode
- Please also refer to methodology.pdf
