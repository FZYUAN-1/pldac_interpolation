import model_abstract as baseM
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
        v_speed = (x_test[0][:2] - x_test[1][:2])/np.maximum(1e-10,(x_test[1][2] - x_test[0][2]))
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

    def toDegrees(self,v):
        return v*180/np.pi   

    def toRadians(self,v):
        return v*np.pi / 180
    
    def toNordBasedHeading(self,GpsHeading):
        return 90 - GpsHeading

    def predictFromInstantSpeed(self,x_test, alpha):
        '''
        based on the fact that we are in small distances, we suppose that a cell is a plane
        param: x_test : [[lat,longi,GpsHeading,GpsSpeed]*N], alpha
                d : [[predi_lat, predi_longi]*N]
                formula source:
                https://cloud.tencent.com/developer/ask/152388
        '''
        radius = 6371e3
        N = len(x_test)
        res = np.zeros((N,2))
        for i in range(len(x_test)):
            lat1, lon1 = toRadians(x_test[i,:2])
            d = x_test[i,3]*alpha/radius
            tc = toRadians(toNordBasedHeading(x_test[i,2]))
            lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(tc))
            dlon = np.arctan2(np.sin(tc)*np.sin(d)*np.cos(lat1), np.cos(d) - np.sin(lat1)*np.sin(lat2))
            lon2= (lon1-dlon + np.pi) % (2*np.pi) - np.pi
            res[i] = [lat2,lon2] 
        return toDegrees(res)
    
    def score(self,func,x_test):
        '''
        param: x_test : [[[prevlat,prevlongi,GpsTime],[lat,longi,GpsTime],[nextlat,nextlongi,GpsTime]]*N]
               func : fonction d'evaluation, i.e. moindre_c
        '''
        #print(x_test[:,2][:,2], x_test[:,1][:,2])
        alpha = x_test[:,2][:,2] - x_test[:,1][:,2]
        
        X_predit = self.predict(x_test[:,:2], alpha)
        X_true = x_test[:,2][:,:2]
        #print(X_predit, X_true)
        return func(X_predit,X_true)

    