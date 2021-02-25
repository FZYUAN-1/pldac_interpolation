import model_abstract as baseM
import numpy as np
from pipeline import *


def strategy_last2(x_train_i, alpha):
    # prendre les 2 derniers points precedents
    return self.predictOne([x_train_i[-2],x_train_i[-1]], alpha)

def strategy_avg(x_train_i, alpha):
    # prendre les moyennes des positions predictes
    return np.mean([self.predictOne([[x_train_i[m], x_train_i[n]]], alpha) for m in range(len(x_train_i)) for n in range(len(x_train_i-1))])

def strategy_rand2(x_train_i, alpha):
    pass

#### Modele Physique
class modele_physique1(baseM.base_model):

    def __init__(self,freq,strategy,eval):
        self.freq = freq
        self.strategy = strategy
        self.eval = eval

    def fit(self, tr):
        '''
        param : tr : dataframe in one cell
                self.x_train : [[[lat1,longi1,GpsTime1],[lat2,longi2,GpsTime2],...,[latN,longiN,GpsTime2]]*N_trip]
        '''
        mat = []
        groups = tr.groupby('Trip')
        for t in groups:
            trip_i = t[1][['Latitude', 'Longitude', 'GpsTime']].to_numpy()
            # take continuous points, step = freq%200
            tmp = []
            for i in range(0,trip_i,int(freq%200)):
                tmp.append(trip_i[i])

            mat.append(tmp)

        mat = np.array(mat)
        
        self.x_train = np.array(mat)
    
    def predictOne(self, x_test, alpha):
        '''
        param: x_test : [[prevlat,prevlongi,GpsTime],[lat,longi,GpsTime]]
        retourne  [nextlat, nextlongi] : prochaine lat et longi apres la duree alpha
        '''
        if x_test[1][2] == x_test[0][2]:
            #print(x_test)
            raise ValueError("GpsTime Equal Error")

        v_speed = (x_test[0][:2] - x_test[1][:2])/(x_test[1][2] - x_test[0][2])
        return x_test[1][:2] + v_speed*alpha

    def predict(self, alpha,strategy):
        '''
        param: [alpha1, alpha2,...,] <=> GpsTime
        retourne  [[nextlat, nextlongi]*N_trip] : prochaine lat et longi apres la duree alpha
        ''' 
        N = len(self.x_train)
        d = np.zeros((N,2))
        for i in range(N):
            d[i] = strategy(self.x_train[i], alpha[i])
        return d

    def score(self,x_test):
        '''
        param: x_test : [[[prevlat,prevlongi,GpsTime],[lat,longi,GpsTime],[nextlat,nextlongi,GpsTime]]*N]
               func : fonction d'evaluation, i.e. moindre_c

               x_test : [[nextlat,nextlongi,GpsTime]*N_trip]
        '''
        #print(x_test)
        
        # alpha = GpsTime_A_n+1 - GpsTime_A_n
        alpha =  x_test[:,2] - self.x_train[:,-1]
        X_predit = self.predict(alpha)
        X_true = x_test[:,:2]
        print(X_true, X_predit)
        return self.eval(X_predit,X_true)
    
