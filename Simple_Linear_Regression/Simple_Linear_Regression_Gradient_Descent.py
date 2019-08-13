import  numpy as np
from tqdm import  tqdm
import matplotlib.pyplot as plt

class Gradient_approach:
    def __init__(self,start_point):

        "start point is a vector contains the slope and the intercept of shape 2*1 "

        self.__start_point = start_point

        return
    def d_intercept(self,X,Y):

        d = -2 * np.sum((Y-(self.__start_point[0]*X)-self.__start_point[1]),axis=0)
        return d

    def d_slope(self,X,Y):

        d = -2 * np.sum(X*(Y-(self.__start_point[0]*X)-self.__start_point[1]),axis=0)
        return d

    def gradient(self,X,Y):

        delta = np.array([self.d_slope(X,Y),self.d_intercept(X,Y)])

        return np.reshape(delta,(2,1))

    def cost(self,observations, predictions):

        e = (observations - predictions)**2
        j = np.sum(e,axis=0)

        return j
    def calc_intercept_slope(self,X,Y,epochs=1000):

        costs = []
        X_mean =  np.mean(X,axis=0)
        X_std =  np.std(X,axis=0)
        X = X - X_mean
        X = X / X_std
        alpah = 0.5
        step_size = alpah/1

        for i in tqdm(range(epochs)):
            delta = self.gradient(X,Y)
            self.__start_point = self.__start_point - (step_size*delta)
            step_size = alpah/(i+2)
            predctions = X*self.__start_point[1] + self.__start_point[0]

            cost = self.cost(Y,predctions)
            costs.append(cost)

        plt.plot([i for i in range(len(costs))],costs)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.show()
        return self.__start_point