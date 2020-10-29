"""
MSc Thesis Quantitative Finance
Title: Interest rate risk simulation using TimeGAN 
       after the EONIA-€STER transition
Author: Lars ter Braak (larsterbraak@gmail.com)

Last updated: October 23rd 2020
Code Author: Lars ter Braak (larsterbraak@gmail.com)

-----------------------------

Inputs
Short rate time series

Outputs
VaR(99%) based on Linear Regression estimation of 1-factor Vasicek for 20 day prediction
"""
    
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import t as tdistr

# 2. Vasicek model calibration using Linear Regression
def model_calibration(data):
    model = LinearRegression().fit(data[:-1].reshape(-1,1), data[1:].reshape(-1,1))
    errors = data[:-1] - (model.intercept_ + model.coef_ * data[1:])
    
    # Computation of model parameters
    k = -np.log(model.coef_)
    mu = model.intercept_ / (1 - model.coef_)
    sigma = np.std(errors) * np.sqrt((- 2 * np.log(model.coef_) ) / (1 - model.coef_**2))
    
    return [k, mu, sigma]

# 3. Value-at-Risk calculation
def VaR(r_0, data, time, percentile=0.99, upward=True):
    # Model calibration
    k, mu, sigma = model_calibration(data)
    
    # VaR calculation
    expectation = r_0 * np.exp(-k*time) + mu * (1 - np.exp(-k * time))
    variance = (sigma**2 / 2*k)*(1 - np.exp(-2*k*time))
    
    # Estimate the degrees of freedom for the t-distribution
    t_df = tdistr.fit(data)[0]
    
    if upward:
        return np.ravel(expectation + np.sqrt(variance) * tdistr.ppf(percentile, t_df))
    else:
        return np.ravel(expectation - np.sqrt(variance) * tdistr.ppf(percentile, t_df))