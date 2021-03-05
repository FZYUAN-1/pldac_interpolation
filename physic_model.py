import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import dataSource as ds
import Eval as ev

class physic_model(BaseEstimator,RegressorMixin):
    def __init__(self, step_train, step_test):
        super().__init__()
        self.v = []
        self.step_train = step_train
        self.step_test = step_test 
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
        #print('shapeX',X.shape)
        for trip in range(len(X)):
            X_t = X[trip]
            y_t = y[trip]
            l_v = []
            differ = y_t - X_t
            #print('d',X_t.shape,y_t.shape)

            for i in range(len(differ)):
                l_v.append([differ[i][0]/differ[i][2], differ[i][0]/differ[i][2], X_t[i][2]])
                
            self.v.append(l_v)

        self.ecart = differ[0][2] / self.step_train
        
        return self

    
    def getInterval(self,x,trip):
        """
        x : [Lat,Lon,GpsTime]
        trouver l'intervalle de GPSTime qu'il correspond
        """
        t = x[-2]
        v_t = self.v[trip]
        for i in range(len(v_t)):
            if v_t[i][2] > t:
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
        for trip in range(len(X)):
            X_t = X[trip]
            v_t = self.v[trip]
            res_t = []
            for i in range(len(X_t)):
                x = X_t[i]
                indice = self.getInterval(x,trip)
                vi = np.array(v_t[indice])
                res_t.append(x[:2] + duration*vi[:2])
            
            res.append(np.array(res_t))

        return res

def moindre_c(X_predit, X_test):
    return ((X_predit-X_test)**2).sum()


freq_train = 1000
freq_test = 400
attrs_x = ['Latitude','Longitude','GpsTime']
labels = ['Latitude','Longitude','GpsTime']

df = ds.importData()
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(df)
pos = [4,4]
tr = ds.trouve_data_case(df, pos, latitude_min,
                         longitude_min, ecart_x, ecart_y)

X_train, X_test, y_train, y_test = ds.train_test_split(df,attrs_x, labels,freq_train,freq_test)
model1 = physic_model(freq_train , freq_test)
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
score = np.mean([moindre_c(yp,yt) for (yp,yt) in zip(y_pred, y_test)])

print("y", score)

"""models = [physic_model(freq_train , freq_test)]

traitement = ev.Traitement(tr,attrs_x,labels)

traitement.set_data_train_test()

#Apprentissage des modèles et évaluation à partir de l'objet traitement
evaluateur = ev.Evaluation(models,traitement)
evaluateur.fit()


res_pred = evaluateur.predict(X_test)

#Affichage des résultats
evaluateur.afficher_resultats()"""