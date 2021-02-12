import models.model_abstract as baseM
import numpy as np

#### Modele Physique
class modele_physique(baseM.base_model):

    def fit(self, datax, datay = None):
        pass
    
    def predictOne(self, x_test, alpha):
        '''
        param: x_test : [[prevlat,prevlongi,GpsTime],[lat,longi,GpsTime]]
        retourne  [nextlat, nextlongi] : prochaine lat et longi apres la duree alpha
        '''
        v_speed = (x_test[0][:2] - x_test[1][:2])/(x_test[1][2] - x_test[0][2])
        return x_test[1][:2] + v_speed*alpha

    def predict(self, x_test, alpha):
        '''
        param: x_test : [[[prevlat,prevlongi,GpsTime],[lat,longi,GpsTime]]*N], [alpha_0,...,alpha_n]
        retourne  [[nextlat, nextlongi]*N] : prochaine lat et longi apres la duree alpha
        ''' 
        N = len(x_test)
        d = np.zeros((N,2))
        for i in range(len(x_test)):
            d[i] = self.predictOne(x_test[i],alpha[i])
        return d
    
    def score(self,func,x_test):
        '''
        param: x_test : [[[prevlat,prevlongi,GpsTime],[lat,longi,GpsTime],[nextlat,nextlongi,GpsTime]]*N]
               func : fonction d'evaluation, i.e. moindre_c
        '''
        print(x_test)
        alpha = x_test[:,2][:,2] - x_test[:,1][:,2]
        X_predit = self.predict(x_test[:,:2], alpha)
        X_true = x_test[:,2][:,:2]
        print(X_true, X_predit)
        return func(X_predit,X_true)

    