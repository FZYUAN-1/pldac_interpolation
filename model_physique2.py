import numpy as np

#### Modele Physique
class model_physique2():
    # predict from instant speed
    # x = [[Trip, Lati, Longi, GpsHeading, GpsSpeed]]

    def __init__(self,freq,coef=0.2,nbr_points=10):
        self.freq = freq
        self.l_freq = [self.freq*coef*i for i in range(nbr_points)]
        self.alpha = None

    def fit(self, x_train, y_train):
        """
        x_train = [[Trip, Lati, Longi, GpsHeading, GpsSpeed]]
        y_train = [[Lati, Longi]]
        """

        radius = 6371e3
        res = None
        erreur = np.inf
        l_trips = np.unique(x_train['Trip'])
        x_trips = x_train.sort_values('Trip')

        for freq in self.l_freq:
            k = 0
            for t in l_trips:
                test = x_trips[x_trips['Trip']==t].sort_values('GpsTime')[['Latitude', 'Longitude','GpsHeading','GpsSpeed']]
                N = test.shape[0]
                tmp = np.zeros((N,2))
                tmp[0,:] =test.iloc[0][['Latitude', 'Longitude']].to_numpy()

                for i in range(N-1):
                    
                    lat1, lon1 = self.toRadians(test.iloc[i,:2])
                    d = test.iloc[i,3]*freq*1e-3/radius
                    tc = self.toRadians(self.toNordBasedHeading(test.iloc[i,2]))
                    lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(tc))
                    dlon = np.arctan2(np.sin(tc)*np.sin(d)*np.cos(lat1), np.cos(d) - np.sin(lat1)*np.sin(lat2))
                    lon2= (lon1-dlon + np.pi) % (2*np.pi) - np.pi
                    tmp[i+1,0] = self.toDegrees(lat2)
                    tmp[i+1,1] = self.toDegrees(lon2)

                if k == 0:
                    res = tmp
                else:
                    res = np.vstack((res,tmp))
                
                k += 1
            tmp_e = self.score(res, y_train)

            if tmp_e < erreur:
                erreur = tmp_e
                self.alpha = freq
        
        print("freq en entrée : ", self.freq, "le meilleur alpha trouvé : ", self.alpha)
        return self.alpha

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
        l_trips = np.unique(x_test['Trip'])
        x_trips = x_test.sort_values('Trip')
        k = 0
        for t in l_trips:
            test = x_trips[x_trips['Trip']==t].sort_values('GpsTime')[['Latitude', 'Longitude','GpsHeading','GpsSpeed']]
            N = test.shape[0]
            tmp = np.zeros((N,2))
            tmp[0,:] =test.iloc[0][['Latitude', 'Longitude']].to_numpy()

            for i in range(N-1):
                
                lat1, lon1 = self.toRadians(test.iloc[i,:2])
                d = test.iloc[i,3]*self.alpha*1e-3/radius
                tc = self.toRadians(self.toNordBasedHeading(test.iloc[i,2]))
                lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(tc))
                dlon = np.arctan2(np.sin(tc)*np.sin(d)*np.cos(lat1), np.cos(d) - np.sin(lat1)*np.sin(lat2))
                lon2= (lon1-dlon + np.pi) % (2*np.pi) - np.pi
                tmp[i+1,0] = self.toDegrees(lat2)
                tmp[i+1,1] = self.toDegrees(lon2)

            if k == 0:
                res = tmp
            else:
                res = np.vstack((res,tmp))
            
            k += 1

        return res

    def score(self,yhat,y):

        return np.sqrt(np.sum((yhat - y.to_numpy()) ** 2))
        