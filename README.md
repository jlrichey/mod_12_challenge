# Module 12 Challenge - Credit Risk Classification

## Overview

In this challenge, I assume the role of growth analyst for MercadoLibre, the largest e-commerce site in Latin America. A Jupyter notebook was produced to import, prepare, analyze and visualize time series company data for the sake of making marketing decisions and revenue forecasts. Three primary data sources were imported that contain Google hourly search trends, closing stock price movement, and daily revenue for MercadoLibre during the study period. 

## Technical Details

The notebook [notebook](forecasting_net_prophet.ipynb), which was developed in Google Colab, loads the following libraries and dependencies.

```python
import numpy as np
import pandas as pd
import holoviews as hv
from prophet import Prophet
import hvplot.pandas
import datetime as dt
%matplotlib inline
```

The data for this project was imported from csv files utilizing the Pandas `.read_csv` method and returned into DataFrames for analysis.  

The following observations were made, and conclusions were drawn:

#### search traffic trends

* The May 2020 Google search traffic for MercadoLibre is 3008.5 above the overall monthly median.
* The "Average Search Traffic by Week of the Year" graph shows a significant drop in search traffic from week 39 to 42. From week 42 to 51 there is a notable increase in traffic. Week 52 illustrates a severe dropoff in average search traffic.
* Midnight is the peak of popularity.
* Tuesday gets the most search traffic for the week.
* Mid October is the lowest point for search traffic in the year.

#### stock price

* Both the "closing price" and "search trend" charts move in tandem to demonstrate a significant drop in late February/early March of 2020. The closing price bottoms out at its lowest point on March 17 and then runs a sideways pattern until it begins to climb on April 1, achieving new highs in May. The search trends continue achieving lower highs than before the late February/early March event, with the exception of a May 5th anomaly.
* The correlation between the "lagged search trends" and the "stock volatility" is approximately -0.15, so there does not seem to be much of a predictable relationship, only a slight negative correlation. The correlation between the "lagged search trends" and the "hourly stock return" is approximately 0.02, which is also not very high.

#### forecasting and predictions (Prophet)

* The near-term forecast for the popularity of MercadoLibre shows a slight dip and then recovery in early to mid-October of 2020.

* Revenue predictions forecast:
  * Most Likely Revenue Forecast: 1,945.45 million,
  * Worst Case Revenue Forecast: 1,772.18 million,
  * Best Case Revenue Forecast: 2,115.97 million.





















## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).



The purpose of this analysis is to build a model, utilizing Logistic Regression, that can correctly identify the creditworthiness of borrowers based on a dataset of historical lending, i.e. `lending_data.csv`.

The dataset includes the following input features (X):
* loan_size
* interest_rate
* borrower_income
* debt_to_income
* num_of_accounts
* derogatory_marks
* total_debt

The output target (y) will be:
* loan_status

The `value_counts()` of the target, `loan_status` are the following. 

<img src="images/value_counts.png" alt="drawing" width="600"/>





## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.








## Sources

The following sources were consulted in the completion of this project. 

* [Holoviews Plotting Documentation](https://hvplot.holoviz.org/)
* [pandas.Pydata.org API Reference](https://pandas.pydata.org/docs/reference/index.html)
* UCB FinTech Bootcamp instructor-led coding exercises


