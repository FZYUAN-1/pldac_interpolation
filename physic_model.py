import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import dataSource as ds
import Eval as ev

class physic_model(BaseEstimator,RegressorMixin):
    def __init__(self, freq):
        super().__init__()
        self.v = {}
        self.freq = freq

    def fit(self, X, y):
        '''
        caculate parameters vector speed self.v for one trajectory by taking the last 2 data points
        Parameters
        ----------
        X : previous data points [[Trip,Lat,Lon,GpsTime]*m]
        y : current data points  [[Trip,Lat,Lon,GpsTime]*m]
        '''
        groups = tr.groupby('Trip')
        for t in groups:
            trip_i = t[1][['Latitude', 'Longitude', 'GpsTime']].to_numpy()
            # take continuous points, step = freq//200
            tmp = []
            for i in range(0,len(trip_i),int(self.freq//200)):
                tmp.append(trip_i[i])

            mat.append(tmp)

        mat = np.array(mat)


        a,b = y[-1],X[-1]
        v_speed = (b[:2] - a[:2])/(a[-1] - b[-1])
        self.v = v_speed
        return self

    def predict(self,X):
        '''
        predict next position

        Parameters
        ----------
        X : [[Trip,Lat,Lon,Duration]*M]
        Duration is the diff between GpsTime of y_test and that of x_test
        '''
        M,N = X.shape
        res = np.zeros((M,2))
        for i in range(M):
            res[i] = self.v*X[i,2] + X[i,:2]
        return res


freq = 400
step = freq//200
attrs_x = ['Latitude','Longitude','GpsTime']
labels = ['Latitude','Longitude']

df = ds.importData()
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(df)
pos = [4,4]
tr = ds.trouve_data_case(df, pos, latitude_min,
                         longitude_min, ecart_x, ecart_y)

df = np.array(mat)

models = [physic_model]

traitement = ev.Traitement(df,attrs_x,labels)

traitement.set_data_train_test(ds.create_data_xy, step, test_size=0.2, random_state=0)

#Apprentissage des modèles et évaluation à partir de l'objet traitement
evaluateur = ev.Evaluation(models,traitement)
evaluateur.fit()


res_pred = evaluateur.predict()

#Affichage des résultats
evaluateur.afficher_resultats()