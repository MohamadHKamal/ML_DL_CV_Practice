import numpy as np
import  Data_Loader

class Simple_Regressio_Analytical_approach:

    def __init__(self):
        return

    def calc_simple_regression_parameters(self,X,Y):

        'X belongs to R-(feature_dims,features_num), Y belongs to R-(observation_dims, features_num) \
        feature_dims = observation_dims'
        X_mean = np.mean(X,axis=0)
        Y_mean = np.mean(Y,axis=0)
        slope = np.sum((X-X_mean)*Y,axis=0) / np.sum((X-X_mean)**2,axis=0)
        intercept = Y_mean - slope * X_mean

        return slope, intercept