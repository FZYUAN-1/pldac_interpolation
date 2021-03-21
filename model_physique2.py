import numpy as np

#### Modele Physique
class modele_physique2():
    # predict from instant speed
    # x = [[Trip, Lati, Longi, GpsHeading, GpsSpeed]]

    def __init__(self,freq):
        self.freq = freq

    def fit(self, x_train, y_train):
        
        return self

    def toDegrees(self,v):
        return v*180/np.pi   

    def toRadians(self,v):
        return v*np.pi / 180
    
    def toNordBasedHeading(self,GpsHeading):
        return 90 - GpsHeading

    def predict(self, x_test):
        '''
        based on the fact that we are in small distances, we suppose that a cell is a plane
        param:  alpha
                d : [[predi_lat, predi_longi]*N]
                formula source:
                https://cloud.tencent.com/developer/ask/152388
        '''
        radius = 6371e3
        res = None
        x_trips = x_test.groupby('Trip')
        k = 0
        for t in x_trips:
            test = t[1][['Latitude', 'Longitude','GpsHeading','GpsSpeed']].to_numpy()

            N = test.shape[0]
            tmp = np.zeros((N,3))
            tmp[:,0] = t[0]
            tmp[0,1:] = test[0,:2]

            for i in range(N-1):
                lat1, lon1 = self.toRadians(test[i,:2])
                d = test[i,3]*self.freq/radius
                tc = self.toRadians(self.toNordBasedHeading(test[i,2]))
                lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(tc))
                dlon = np.arctan2(np.sin(tc)*np.sin(d)*np.cos(lat1), np.cos(d) - np.sin(lat1)*np.sin(lat2))
                lon2= (lon1-dlon + np.pi) % (2*np.pi) - np.pi
                tmp[i+1,1] = self.toDegrees(lat2)
                tmp[i+1,2] = self.toDegrees(lon2)

            if k == 0:
                res = tmp
            else:
                res = np.vstack((res,tmp))
            
            k += 1
        return res[:,1:]

    def score(self,x_test,y_test):
        predict = self.predict(x_test)
        
        return np.sqrt(np.sum((y_test.to_numpy() - predict) ** 2))
        