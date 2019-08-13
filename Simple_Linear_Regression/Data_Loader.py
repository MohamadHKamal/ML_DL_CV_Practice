import numpy as np
import pandas as pd
import copy

'I will include a new methods to this file incrementally depends on need'

class Regression_DataLoader:

    def __init__(self, csv_path, num_dim , features_cols=[''], targets_cols=[''],
                 numerical_values={'cols_name':[('not_numerical_value_name',0)]} ):

        "num_dim = len(features_cols)+len(targets_cols), features_cols is list of strings contains the columns names in \
        csv file that is used as features, targets_cols contains the columns names of cvs file that is used as observations\
        numerical_values is a dictionary that contains columns name of csv file (contains non numerical values) as a key\
        and a list of ordered pair as a value where the first element is the non numerical value and the second is the corresponding\
        numerical value if your features_cols and targets_cols take numerical values then you must pass numerical_values as {} "

        if num_dim != len(features_cols)+len(targets_cols):
            print('Error: num_dim != len(features_cols)+len(targets_cols) occurred at the initializer of Regression_DataLoader ')
        self.__csv_path = copy.copy(csv_path)
        self.__features_cols = copy.copy(features_cols)
        self.__target_cols = copy.copy(targets_cols)
        self.__numerical_values = copy.copy(numerical_values)
        self.__flag_non_numerical_values = True

        if len(numerical_values.keys()) == 0:
            self.__flag_non_numerical_values = False
        else:
            print("The case of non numerical data not implemented yet !")
        return

    def load_data(self):
        "This function load the features and the observations from the csv file into features_mat and observations_mat\
        features_mat belongs to R-(feature_dims,nums_features) and observations_mat belongs to R-(observation_dims,nums_features)"

        csv_file = pd.read_csv(self.__csv_path)
        features_mat = np.array([csv_file[features_col] for features_col in self.__features_cols])
        observations_mat = np.array([csv_file[observations_col] for observations_col in self.__target_cols])

        return features_mat.T, observations_mat.T

