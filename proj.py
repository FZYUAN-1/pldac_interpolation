# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
def readfile(filename_csv, n = 1000000):
    df = pd.read_csv(filename_csv, nrows = n)
    df = df[(df["Latitude"] >= 42.282970-0.003) & (df["Latitude"] <= 42.282970+0.003) & (df["Longitude"] >= -83.735390-0.003) & (df["Longitude"] <= -83.735390+0.003)]
    npTrips, counts = np.unique(np.array(df['Trip']), return_counts=True)
    npTrips = npTrips[counts>100]
    df = df[df['Trip'].isin(npTrips)]

    print(df.shape)
    # print(csv_data)
    return df

# %%
df = readfile('dataset/DataGpsDas.csv')

# %%
print(df)
 # %%   
def plot(dataframe):
    f = dataframe[['Trip', 'Latitude', 'Longitude']]
    plt.scatter(f['Longitude'], f['Latitude'])
    plt.show()
    

plot(df)
    
# %%
# modele physique
def nextpos(pos, alpha, v):
    return pos + alpha*v
# %%
datax = df.to_numpy()[:,5:7]
# %%
def split_plan(df, n_interval):
    data = df.to_numpy()
    # unpack dimension
    M, N = n_interval
    ub_x = data[:,1].max()
    lb_x = data[:,1].min()
    gap_x = (ub_x-lb_x)/M
    ub_y = data[:,2].max()
    lb_y = data[:,2].min()
    gap_y = (ub_y-lb_y)/N
    intervals_x = [lb_x + gap_x*k for k in range(1,M)]
    intervals_y = [lb_y + gap_y*k for k in range(1,N)]
    return intervals_x, intervals_y


# %%
def affectation(datax, intervals_x, intervals_y):
    M, N = datax.shape
    for i in range(M):
        x, y = datax[i,:]
        datax[i] = np.argmax(intervals_x>x), np.argmax(intervals_y>y)
    return datax
# %%
inter_x, inter_y = split_plan(df, (1000,2000))
raw_datay = affectation(datax, inter_x, inter_y)
# %%
def to_labeled_points(raw_datay):
    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #  rest, droite, hautdroite, haut, hautgauche, gauche, basgauche, bas, basdroite
    M = len(raw_datay)
    datay = np.zeros(M)
    for i in range(M):
        diff = (raw_datay[i+1] - raw_datay[i])
        if diff[0]>0:
            if diff[1]>0:
                grad = diff[1]/diff[0]
                if grad > 1:
                    data[y] = 
            
        else:

        datay[i] = datay[i,1]/datay[i,0]

    return datay

# %%
