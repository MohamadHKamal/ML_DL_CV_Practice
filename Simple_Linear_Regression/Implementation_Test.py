import Data_Loader
import Simple_Linear_Regression_Analytical
import Linear_Regression_sklearn
import numpy as np
import Simple_Linear_Regression_Gradient_Descent
if __name__ == '__main__':

    Data_obj = Data_Loader.Regression_DataLoader('Philadelphia_Crime_Rate_noNA.csv', 2, ['CrimeRate'], ['HousePrice'], {})
    Features_mat, Observations_mat = Data_obj.load_data()
    analytical_obj = Simple_Linear_Regression_Analytical.Simple_Regressio_Analytical_approach()
    sk_api_obj = Linear_Regression_sklearn.sk_API_Regression()
    analytical_slope,analytical_intercept = analytical_obj.calc_simple_regression_parameters(Features_mat, Observations_mat)
    sk_api_slope,sk_api_intercept = sk_api_obj.calc_simple_regression_parameters(Features_mat, Observations_mat)
    np.testing.assert_almost_equal(sk_api_slope,analytical_slope[0])
    np.testing.assert_almost_equal(sk_api_intercept,analytical_intercept)

    print('Analytical approach succeed')

    start_point = np.zeros((2,1))
    gradient_obj = Simple_Linear_Regression_Gradient_Descent.Gradient_approach(start_point)
    components = gradient_obj.calc_intercept_slope(Features_mat,Observations_mat)
    slope = [components[0]]
    intercept = [components[1]]
    np.testing.assert_almost_equal(sk_api_slope, slope)
    np.testing.assert_almost_equal(sk_api_intercept, intercept)


    print('Gradient approach succeed')