import model_abstract as baseM
import numpy as np

#### Modele Physique
class modele_physique2(baseM.base_model):
    # predict from instant speed

    def fit(self, x_train, datay = None):
        # param: x_test : [[lat,longi,GpsHeading,GpsSpeed]*N]
        self.x_train = x_train

    def toDegrees(self,v):
        return v*180/np.pi   

    def toRadians(self,v):
        return v*np.pi / 180
    
    def toNordBasedHeading(self,GpsHeading):
        return 90 - GpsHeading

    def predict(self, alpha):
        '''
        based on the fact that we are in small distances, we suppose that a cell is a plane
        param:  alpha
                d : [[predi_lat, predi_longi]*N]
                formula source:
                https://cloud.tencent.com/developer/ask/152388
        '''
        radius = 6371e3
        N = len(self.x_train)
        res = np.zeros((N,2))
        for i in range(N):
            lat1, lon1 = self.toRadians(self.x_train[i,:2])
            d = self.x_train[i,3]*alpha/radius
            tc = self.toRadians(self.toNordBasedHeading(self.x_train[i,2]))
            lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(tc))
            dlon = np.arctan2(np.sin(tc)*np.sin(d)*np.cos(lat1), np.cos(d) - np.sin(lat1)*np.sin(lat2))
            lon2= (lon1-dlon + np.pi) % (2*np.pi) - np.pi
            res[i] = [lat2,lon2] 
        return self.toDegrees(res)

    def score(self,func,x_test):
        pass