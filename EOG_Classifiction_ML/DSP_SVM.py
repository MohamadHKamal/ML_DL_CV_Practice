from sklearn import svm
import numpy as np
class Hero_SVM:
    def Create_Model(self,kernel='linear', c=300, gamma=0.0001):
        "This function create a model with the given hyper parameters \
        and return the model"
        Model = svm.SVC(kernel=kernel, C=c, gamma=gamma, probability=True)
        return Model

    def Train_Model(self,Features, lbls, Model):
        "This function train the given model on the features data and its labels \
         and return the model after train."
        Model.fit(Features, lbls)
        return Model

    def Calculate_Accuracy(self,Features_Data, lbls, Model):
        "This function calculate the accuracy and return it"

        Acc = Model.score(Features_Data, lbls)

        return Acc

    def Predict_lbls(self,Features_Data, Model):
        "This function predict the classes of the given features by the given model and return the labels"

        lbls = Model.predict_proba(Features_Data)

        return lbls

