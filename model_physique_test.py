# %%
# Importation des librairies
import model_physique as m_phy
import methods as mtds
import dataSource as ds
import dataVisualization as dv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%
# Récupération des données et des paramètres
df = ds.importData()
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = ds.calcul_param(
    df)
pos = (4, 4)
n_interval = 10

# %%
# group by trip, convert to numpy matrix
# mat = [[[data points in one trip]] * nb_trip], data points : [lat, lon, GpsTime]
mat = []


tr = ds.trouve_data_case(df, pos, latitude_min,
                         longitude_min, ecart_x, ecart_y)
# print(tr.shape)
for t in tr.groupby('Trip'):
    mat.append(t[1][['Latitude', 'Longitude', 'GpsTime']].to_numpy())


# %%
# Création instance du modele physique
m = m_phy.modele_physique()

# %%


def take_continuous(pts, start, stop, step, n):
    res = []
    while start < stop:
        if len(res) >= n:
            break
        res.append(pts[start])
        start += step
    return res


# Fonctions splits

def split_OneTrip(trip, take_function, st=0.5, k1=0.1, k2=0.1, random=0):
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
    if random:
        # prev_points = trip[np.random.choice(N, np.max(int(N*k1)), 1)]
        # scored_points = trip[np.random.choice(
        #    np.arange(N, len(trip)), np.max(int((len(trip)-N)*k2)), 1)]
        pass
    else:
        return take_function(trip, 0, N, 1, max(int(N*k1), 1)), take_function(trip, N, len(trip), 1, max(int((len(trip)-N)*k2), 1))


def split_AllTrip(trips, func):
    first=[]
    last=[]
    for trip in trips:
        tmp1, tmp2=split_OneTrip(trip, func)
        first.append(tmp1)
        last.append(tmp2)
    return first, last

# %%
# Fonction test

# Test for (A_n-1, A_n, A_n+1) physics model prediction
def test1(trips, func_cost):
    mat_first, mat_last=split_AllTrip(trips, take_continuous)
    cost=0
    # print(mat_first, mat_last)
    for i in range(len(mat_first)):
        first, last=mat_first[i], mat_last[i]
        # print(first, last)
        # returns all combinations of tuple (A_n-1, A_n, A_n+1)
        for elem_last in last:
            for j in range(len(first)-1):
                for k in range(j+1, len(first)):
                    cost += m.score(func_cost,
                                    np.array([[first[j], first[k], elem_last]]))

    return cost

# Test for prediction from GpsHeading and GpsSpeed
def test2(trips, func_cost):
    mat_first, mat_last=split_AllTrip(trips, take_continuous)
    cost=0
    # print(mat_first, mat_last)
    for i in range(len(mat_first)):
        first, last=mat_first[i], mat_last[i]
        # print(first, last)
        # returns all combinations of tuple (A_n, A_n+1)
        for elem_last in last:
            for j in range(len(first)):
                    cost += m.score2(func_cost,
                                    np.array([[first[j], elem_last]]))
    return cost


# %%
# Calcul de la matrice des erreurs


def calcul_mat_err(df, latitude_min, longitude_min, ecart_x, ecart_y, n_interval=10):
    """ DataFrame * float * float * float * flaot * int -> list(list(float))

            Retourne une matrice contenant l'erreur à chaque case.
    """
    # matrice erreur des cases
    mat_err=np.zeros((n_interval, n_interval))
    # parcours de toutes les cases
    for i in range(n_interval):
        for j in range(n_interval):
            # récupération des données de la case
            case_df=ds.trouve_data_case(
                df, (i, j), latitude_min, longitude_min, ecart_x, ecart_y)
            if case_df.shape[0] > 0:
                # erreur des trips sur la case
                mat=[]
                for t in case_df.groupby('Trip'):
                    mat.append(
                        t[1][['Latitude', 'Longitude', 'GpsTime']].to_numpy())

                mat_err[n_interval-1-i, j]=test1(mat, mtds.moindre_c)
                #print(mat_err[n_interval-1-i, j])

    return mat_err


# %%
# Test

mat_err = calcul_mat_err(df, latitude_min, longitude_min, ecart_x, ecart_y)
dv.afficher_mat(mat_err,"Matrice erreur 1", n_interval)
dv.afficher_mat_hist(mat_err)

# %%

def calcul_mat_err2(df, latitude_min, longitude_min, ecart_x, ecart_y, n_interval=10):
    """ DataFrame * float * float * float * flaot * int -> list(list(float))

            Retourne une matrice contenant l'erreur à chaque case.
    """
    # matrice erreur des cases
    mat_err=np.zeros((n_interval, n_interval))
    # parcours de toutes les cases
    for i in range(n_interval):
        for j in range(n_interval):
            # récupération des données de la case
            case_df=ds.trouve_data_case(
                df, (i, j), latitude_min, longitude_min, ecart_x, ecart_y)
            if case_df.shape[0] > 0:
                # erreur des trips sur la case
                mat=[]
                for t in case_df.groupby('Trip'):
                    mat.append(
                        t[1][['Latitude', 'Longitude', 'GpsHeading', 'GpsSpeed', 'GpsTime']].to_numpy())

                mat_err[n_interval-1-i, j]=test2(mat, mtds.moindre_c)
                #print(mat_err[n_interval-1-i, j])

    return mat_err

#dv.affiche_carte(df, pos, latitude_min, latitude_max,
#                 longitude_min, longitude_max, ecart_x, ecart_y)
# dv.afficher_hist_norm_vit(df, pos, latitude_min, longitude_min, ecart_x, ecart_y)


mat_err=calcul_mat_err2(df, latitude_min, longitude_min, ecart_x, ecart_y)
dv.afficher_mat(mat_err,"Matrice erreur 2",n_interval)
dv.afficher_mat_hist(mat_err)
