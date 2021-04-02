from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, zero_one_loss, mean_squared_error

import numpy as np

class model_contextual_knnReg(BaseEstimator,RegressorMixin):

    def __init__(self):
        super().__init__()
        self.clf = None
        

    def fit(self, X, y):
        '''
        param X [[Latitude,Longitude,GpsHeading,GpsSpeed]*m]
              y [[Latitude,Longitude]*m]
        '''
        params = {'n_neighbors':range(1,10)}
        knr = KNeighborsRegressor()
        clf = GridSearchCV(knr,params)

        #idx = int(len(X)*0.7)
        #X_train, X_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]

        #clf.fit(X_train,y_train)
        clf.fit(X,y)

        print('Les meilleurs paramètres trouvés sont:', clf.best_params_)

        #y_test, y_pred = y_test, clf.predict(X_test)

        #print(classification_report(y_test, y_pred))

        self.clf = clf

        return self

    def predict(self,X):
        '''
        param [[Trip,Latitude,Longitude,GpsTime]*m]
        retourne  [[nextlat, nextlongi]*m] : prochaine lat et longi
        ''' 

        return self.clf.predict(X)
    
    #def score(self,X_test,y_test):
    #    return mean_squared_error(y_test,self.predict(X_test))
