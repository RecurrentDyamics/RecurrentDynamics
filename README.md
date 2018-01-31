# RecurrentDynamics

This is the **anonymouse** repo of IJCAI submission **Imputing Missing Values in Multivariate Time Series with Bidirectional Recurrent Dynamics**.

This repo includes the data descriptions and model implementation codes of our experiment.

## Data Description

### Syntheic Time Series Regression
The data is generated with python script generate.py. Each item contains a time series input $X$, a label for $T+1$ step $y$, and the masking vector $mask$. To use the data, we arrange the time series into a JSON file (see Synthetic/data for samples).

### Airquality
The data can be downloaded at https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip and the detailed data description can be found in https://www.microsoft.com/en-us/research/publication/st-mvl-filling-missing-values-in-geo-sensory-time-series-data/.

See Airquality/data for samples.

### HealthCare Data
The data and detailed description can be found in https://www.physionet.org/challenge/2012/.

See HealthCare/data for samples.

## Code Description
The code for three experiments are almost the same. We provide the code for the airquality experiment. Users can use our model on the other experiments by simply replacing the data file.

See Airquality for details.
