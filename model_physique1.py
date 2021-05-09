from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_squared_error

#### Modele Physique
class model_physique1(BaseEstimator):

    def __init__(self):
        super().__init__()
        #self.step_test = None
        #self.set_params(**{'step_test':step_test})
        self.globaltheta = None

    def fit(self, X, y):
        return self

    def predict(self,X):
        '''
        param [['Trip','Latitude','Longitude','GpsHeading','GpsTime']*m]
        retourne  [[nextlat, nextlongi]*m] : prochaine lat et longi
        ''' 
        
        mat = []
        groups = X.groupby('Trip')
        for group in groups:
            mat.append(group[1][['Latitude','Longitude','GpsTime']].to_numpy())
        
        #print(mat)
        try:
            res = []
            train_step = mat[0][1][2] - mat[0][0][2]
        except IndexError:
            print('indexerror',X, mat)
        #print(train_step)

        for X_t in mat:
            #res.append(X_t[0][:2])

            for i in range(1,len(X_t)):
                v = np.array([ (X_t[i][0] - X_t[i-1][0]) / train_step , (X_t[i][1] - X_t[i-1][1]) / train_step ])
                if self.globaltheta:
                    v = v*self.globaltheta
                
                #print(v)
                #print(self.step_test)
                
                res.append(X_t[i-1][:2] + train_step*v)
        #print(self.step_test)
        #print(len(res), len(res[0]))
        #print(res)
        return np.array(res)

    