
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
#### Modele Physique
class model_physique1bis(BaseEstimator):

    def __init__(self,l_alpha,freq_test,iftheta=False):
        super().__init__()
        self.freq_test = freq_test
        self.l_alpha = l_alpha
        self.iftheta = iftheta
    def fit(self, X, y):

        if self.iftheta:
            self.theta = X['GpsHeading'].mean()
        
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

        try:
            train_step = mat[0][1][2] - mat[0][0][2]
        except IndexError:
            print('indexerror',X, mat)

        ind = self.freq_test//200 - 1
        #print(ind)
        try:
            #print(t)
            alpha = self.l_alpha[int(ind)] 
            
        except IndexError:
            print("Oops!  Index n'est pas dans l_alpha. ", t, ind)

        res = []

        for X_t in mat:
            res.append(X_t[0][:2])

            for i in range(1,len(X_t)):
                v = np.array([ (X_t[i-1][0] - X_t[i][0]) / train_step , (X_t[i-1][1] - X_t[i][1])/ train_step ])
                #print(X_t[i][:2].shape, v.shape, alpha.shape)
                #print(X_t[i][:2].shape, (v@alpha).shape)
                if self.iftheta:
                    theta = self.theta
                    r = np.array(( (np.cos(theta), -np.sin(theta)),
                                (np.sin(theta),  np.cos(theta)) ) )
                    v = v@r
                res.append(X_t[i][:2] + v@alpha)        

        return np.array(res)



def learn_alpha(freq_train, X, y):
    """ 
    X : dim (n,3)
    y : dim (n,3)
    """
    
    # (A' - A)/t, dim (n,2)
    t = X[0,2] - y[0,2]
    X_train = (y[:,:2] - X[:,:2])/t
    
    # -A + y, dim (n,2)
    y_train = y[:,:2] - X[:,:2]
    

    clf = LinearRegression()
    clf.fit(X_train,y_train)

    print('coef: ' ,clf.coef_, clf.coef_.shape)
        
    return clf.coef_

