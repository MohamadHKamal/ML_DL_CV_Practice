from sklearn.neighbors import KNeighborsClassifier
import numpy as np
class KNN():
    def fit(self,Data,lbls,n_neighbors):
        Knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        Knn_classifier.fit(Data,lbls)
        return Knn_classifier


    def calc_Acc(self,Data,lbls,Knn_classifier):
        Acc=0
        predections = Knn_classifier.predict(Data)
        for i in range(len(predections)):
            if predections[i]==lbls[i]:
                Acc=Acc+1
        return (Acc/len(predections))