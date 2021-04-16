
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
#### Modele Physique
class model_physique1bis(BaseEstimator):

    def __init__(self,l_alpha,iftheta=False):
        super().__init__()
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
        train_step = 0
        #som = 0
        for group in groups:
            tmp = group[1][['Latitude','Longitude','GpsTime']].to_numpy()
            mat.append(tmp)
            if tmp.shape[0] >= 2 and train_step == 0:
                temps_ecart = tmp[1:,2] - tmp[:-1,2]
                train_step = np.min(temps_ecart)
                if train_step > 10000:
                    print(temps_ecart)
                 
        
        #print(train_step)
        #train_step /= som
        #print('M',mat)
        """try:
            train_step = mat[0][1][2] - mat[0][0][2]
        except IndexError:
            print('indexerror',X, mat)"""

        ind = train_step//200 - 1
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
                if self.iftheta:
                    theta = self.theta
                    r = np.array(( (np.cos(theta), -np.sin(theta)),
                                (np.sin(theta),  np.cos(theta)) ) )
                    v = v@r
                res.append(X_t[i][:2] + v@alpha)   
                #res.append(X_t[i][:2] + v*train_step)       

        return np.array(res)



def learn_alpha(freq_train, A_pre, A, y):
    """ 
    X : dim (n,3)
    y : dim (n,3)
    """
    # (A' - A)/t, dim (n,2)
    t = A[0,2] - A_pre[0,2]
    X_train = (A_pre[:,:2] - A[:,:2])/t

    #print(A, A_pre)
    
    # -A + y, dim (n,2)
    y_train = - y[:,:2] + A[:,:2]
    

    clf = LinearRegression()
    clf.fit(X_train,y_train)

    print('coef: ' ,clf.coef_, clf.coef_.shape)
        
    return clf.coef_
