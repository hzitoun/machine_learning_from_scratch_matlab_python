# -*- coding: utf-8 -*-
import numpy as np

#https://en.wikipedia.org/wiki/Coefficient_of_determination
def r2_score(y_true, y_pred):
    mean = np.mean(y_true) #mean of the observed dat
    SS_tot = np.sum((y_true - mean) ** 2) #The total sum of squares (proportional to the variance of the data)
    SS_res  = np.sum((y_true - y_pred) ** 2) #The sum of squares of residuals, also called the residual sum of squares
    r2_score = 1 - (SS_res / SS_tot) #The most general definition of the coefficient of determination
    return r2_score

