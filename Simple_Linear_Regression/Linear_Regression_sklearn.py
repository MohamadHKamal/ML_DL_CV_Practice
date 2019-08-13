from sklearn.linear_model import  LinearRegression

class sk_API_Regression:
    def __init__(self):
        self.__Reg_Model = LinearRegression()
        return
    def calc_simple_regression_parameters(self,Features_mat,Observations_mat):
        train_instance = self.__Reg_Model.fit(Features_mat,Observations_mat)
        return train_instance.coef_,train_instance.intercept_

