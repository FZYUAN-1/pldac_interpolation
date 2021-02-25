import model_physique1 as md1

def moindre_c(X_predit, X_test):
    #print(X_predit, X_test)
    return ((X_predit-X_test)**2).sum()

class Pipeline:
    def __init__(self,raw_data_in_one_cell,model):
        self.data = raw_data_in_one_cell
        self.model = model
        #self.eval = eval

    def exec(self):
        self.model.fit(self.data)
        res = self.model.score()
        print(res)
        # visualisation..
        return res

model = md1.modele_physique1(400,md1.strategy_last2,moindre_c)
pipeline = Pipeline()
pipeline.exec()





# Fonctions d'echantillonage pour traitement des donnees
'''
def take_continuous(pts, start, stop, step, n):
    res = []
    while start < stop:
        if len(res) >= n:
            break
        res.append(pts[start])
        start += step
    return res
# Fonctions splits

def echanti_OneTrip(trip, take_function, st=0.5, freq = 400, k1=0.1, k2=0.1, random=0):
    '''
    trip : np.array [data points in one trip]
    st : sample from first st% data_points
    k1: from the first st% data_points, we randomly sample k1% numbers of points
    k2: from the last (1-st)% data_points, we randomly sample k2% numbers of points

    [p1     p2      p3      p4      p5      p6]
                    |               |        |
                    k1              st       k2
    returns : Sampled_prev_points : [[lat,lon,gpstime]*(N*k1)], Sampled_score_points[[lat,lon,gpstime]*(int((len(trip)-N)*k2))]
    '''
    if len(trip) < 3:
        return [], []
    N = int(len(trip)*st)
    else:
        return take_function(trip, 0, N, freq%200, max(int(N*k1), 1)), take_function(trip, N, len(trip), freq%200, max(int((len(trip)-N)*k2), 1))

def echanti_AllTrip(trips, func):
    first=[]
    last=[]
    for trip in trips:
        tmp1, tmp2 = echanti_OneTrip(trip, func)
        first.append(tmp1)
        last.append(tmp2)
    return first, last
'''