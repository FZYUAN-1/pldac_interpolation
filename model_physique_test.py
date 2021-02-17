# %%
import model_physique as m_phy
import methods as mtds
import dataSource as ds
import numpy as np

# %%
df = ds.importData()
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(df)

# %%
# group by trip, convert to numpy matrix
# mat = [[[data points in one trip]] * nb_trip], data points : [lat, lon, GpsTime]
mat = []

# case 2,1
tr = ds.trouve_data_case(df,(2,1),latitude_min,longitude_min,ecart_x,ecart_y)
for t in tr.groupby('Trip'):
    mat.append(t[1][['Latitude','Longitude','GpsTime']].to_numpy())

mat


# %%
m = m_phy.modele_physique()

# %%
def split_OneTrip(trip, st = 0.5, k1 = 1, k2 = 1, random = 1):
    '''
    trip : np.array [data points in one trip]
    st : sample from first st% data_points
    k1: from the first st% data_points, we randomly sample k1% numbers of points
    k2: from the last (1-st)% data_points, we randomly sample k2% numbers of points

    [p1     p2      p3      p4      p5      p6]
            |               |           |   
            k1              st          k2
    ''' 
    prev_points = []
    scored_points = []

    if random:
            N = int(len(trip)*st)
            prev_points = trip[np.random.choice(np.arange(N), int(N*k1))]
            scored_points = trip[np.random.choice(np.arange(N, len(trip)), int((len(trip)-N)*k2))]
            return prev_points, scored_points
    else:
        pass
    

def split_AllTrip(trips, params = None):
    first = []
    last = []
    for trip in trips:
        tmp1, tmp2 = split_OneTrip(trip)
        first.append(tmp1)
        last.append(tmp2)
    return first, last


def test1(trips,func_cost):
    mat_first, mat_last = split_AllTrip(trips)
    cost = 0
    print(mat_first, mat_last)
    for i in range(len(mat_first)):
        first, last = mat_first[i], mat_last[i]
        #print(first, last)
        # returns all combinations of tuple (A_n-1, A_n, A_n+1)
        for elem_last in last:
            for j in range(len(first)-1):
                for k in range(j+1, len(first)):
                    tmp = m.score(func_cost,np.array([[first[j], first[k], elem_last]]))
                    cost += tmp
    return cost


# %%

cost = test1(mat,mtds.moindre_c)

print(cost)

# %%
