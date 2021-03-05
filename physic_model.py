import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import dataSource as ds
import Eval as ev

class physic_model(BaseEstimator,RegressorMixin):
    def __init__(self, step_train, step_test):
        super().__init__()
        self.v = []
        self.step_train = step_train # liste, pour les trips différents
        self.step_test = step_test # liste, pour les trips différents
        self.ecart = None

    def fit(self, X, y):
        '''
        caculate parameters vector speed self.v for one trajectory by taking the last 2 data points
        Parameters
        ----------
        X : previous data points [[Lat,Lon,GpsTime]*m,[]]
        y : current data points  [[Lat,Lon,GpsTime]*m,[]]
        v : [[dLat/t, dLon/t, GpsTime],[]]
        '''

        for trip in len(X):
            X_t = X[trip]
            y_t = y[trip]
            l_v = []
            differ = y_t - X_t

            for i in range(len(differ)):
                l_v.append([differ[i,0]/differ[i,2], differ[i,0]/differ[i,2], X[i,2]])
                
            self.v.apprend(l_v)

        self.ecart = differ[0,2] / self.step_train
        
        return self

    
    def getInterval(self,x):
        """
        x : [Lat,Lon,GpsTime]
        trouver l'intervalle de GPSTime qu'il correspond
        """
        t = x[-2]
        for i in range(len(self.v)):
            if self.v[i][2] > t:
                return i-1
        
        return -1



    def predict(self,X):
        '''
        predict next position

        Parameters
        ----------
        X : [[Lat,Lon,GpsTime]*M]
        On prédict les prochains points de X , Duration = step_test * self.ecart
        '''
        duration = self.step_test * self.ecart

        res = []
        for trip in len(X):
            X_t = X[trip]
            v_t = self.v[trip]
            res_t = []
            for i in range(len(X_t)):
                x = X_t[i]
                indice = self.getInterval(x)
                vi = np.array(v_t[indice])
                res_t.append(x[:1] + duration*vi[:1])
            
            res.append(np.array(res_t))
            
        return np.array(res)



freq = 400
step = freq//200
attrs_x = ['Latitude','Longitude','GpsTime']
labels = ['Latitude','Longitude']

df = ds.importData()
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(df)
pos = [4,4]
tr = ds.trouve_data_case(df, pos, latitude_min,
                         longitude_min, ecart_x, ecart_y)

attrs_x = [['Latitude','Longitude','GpsTime']]
#targets
labels = ['Latitude','Longitude','GpsTime']

models = [physic_model()]

traitement = ev.Traitement(tr,attrs_x,labels)

traitement.set_data_train_test(ds.train_test_split, attrs_x, labels, 1000, 400)

#Apprentissage des modèles et évaluation à partir de l'objet traitement
evaluateur = ev.Evaluation(models,traitement)
evaluateur.fit()


res_pred = evaluateur.predict()

#Affichage des résultats
evaluateur.afficher_resultats()