# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Sujet PLDAC : Interpolation contextualisée
# 
# 
# **Enseignants**: Vincent Guigue & Nicolas Baskiotis
# 
# %% [markdown]
# ## **Contexte :**
# L’idée est de développer des algorithmes prédictifs et d’interpolation sur les trajectoires
# GPS de voiture.<br/>
# En effet, la plupart des traces disponibles sont en très bas échantillonnage... Ce qui bride de nombreuses application.<br/>
# A partir d’un ensemble de données *T* = {x<sub>1</sub>, ... , x<sub>t</sub>, ..  , x<sub>T</sub>} ∈ *R<sup>2×T</sup>*, il est possible de
# chercher à prédire *x<sub>T+1</sub>* : cela constitue un problème classique.<br/> Il est aussi possible de chercher
# à prédire *x<sub>t+1/2</sub>* qui se situe entre les pas de temps *t* et *t+1*.<br/>
# Nous nous intéressons à ces problèmes... Mais en ajoutant un ingrédient important :
# la connaissance du contexte. En effet, à certaines intersections, on peut trouver à partir
# des observations du passé que tout le monde tourne à gauche : sachant cela, les problèmes
# précédents deviennent plus faciles.<br/>
# A partir d’une base des traces GPS haute fréquence sur la ville de Détroit, le but est
# d’apprendre à la fois les modèles de prédiction/interpolation et une représentation du contexte
# spatial.
# 
# ------------
# %% [markdown]
# ## Description rapide
# %% [markdown]
# **Fichier:  DataGpsDas.csv**<br>
# Source : https://catalog.data.gov/dataset/safety-pilot-model-deployment-data <br/>
# 
# -----------------------
# 
# **Nombre de lignes total dans le fichier :** 41,021,227 <br/>
# **Nombre de colonnes :** 17
# 
# ------------------------
# 
# **Nom des colonnes :**   ['Device', 'Trip', 'Time', 'GpsTime', 'GpsWeek', 'GpsHeading',
#        'GpsSpeed', 'Latitude', 'Longitude', 'Altitude', 'NumberOfSats',
#        'Differential', 'FixMode', 'Pdop', 'GpsBytes', 'UtcTime', 'UtcWeek']
# 
# -----------------------
# 
# ### **Description de quelques colonnes :**<br/><br/>
# 
# 
# 
# **Attribute Label:  DeviceID (column 0)** <br/>
# **Attribute Definition:** This field contains the unique, numeric ID assigned to each DAS. This ID also doubles as a vehicle’s ID. <br/>
# **Attribute Domain Values:** Integer
# 
# ------------------
# 
# **Attribute Label: Trip (column 1)** <br/>
# **Attribute Definition:** This field contains a count of ignition cycles—each ignition cycle commences when the ignition is in the on position and ends when it is in the off position. <br/>
# **Attribute Domain Values:** Integer
# 
# ------------------
# 
# **Attribute Label: Time (column 2)** <br/>
# **Attribute Definition:** This field contains the time in centiseconds since DAS started, which (generally) starts when the ignition is in the on position. <br/>
# **Attribute Domain Values:** Integer
# 
# -------------------
# 
# **Attribute Label: GPS_Speed  (column 6)** <br/>
# **Attribute Definition:** This field contains the speed, in meters/second, of vehicle according to GPS. <br/>
# **Attribute Domain Values:** Float
# 
# -------------------
# 
# **Attribute Label: GPS_Latitude (column 7)** <br/> 
# **Attribute Definition:** This field contains the latitude, in degrees, of vehicle according to GPS.<br/> 
# **Attribute Domain Values:** Float
# 
# 
# -------------------
# 
# 
# **Attribute Label: GPS_Longitude (column 8)**<br/>
# **Attribute Definition:** This field contains the longitude, in degrees, of vehicle according to GPS.<br/>
# **Attribute Domain Values:** Float
# 
# -------------------
# 
# **Attribute Label: GPS_Pdop  (column 13)** <br/>
# **Attribute Definition:** This field contains the Positional Dilution of Precision, used to determine position accuracy; the lower the number, the better. <br/>
# **Attribute Domain Values:** Float
# %% [markdown]
# ### Importation de librairies

# %%
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import models.model_physique as mp

# %% [markdown]
# ### Lecture du fichier

# %%
# Lecture du fichier csv : 1,000,000 premières lignes
df = pd.read_csv("../DataGpsDas.csv", nrows=1000000)

# %% [markdown]
# ### Trie des données
# %% [markdown]
# Informations tirées du pdf *Motion ResNet : An efficient data imputation method for spatio-temporal series* :
# 
# 1. "We took trajectories only passing in a defined perimeter centered in latitude 42.282970 and longitude -83.735390, all positions within **latitude 42.282970±0.003000** and **longitude −83.735390±0.003000** are kept."
# 
# 
# 2. "Also we keep only trajectories with **at least 100 data points** so that we have enough dynamics to learn something."

# %%
#1.
df = df[(df["Latitude"] >= 42.282970-0.003) & (df["Latitude"] <= 42.282970+0.003) 
        & (df["Longitude"] >= -83.735390-0.003) & (df["Longitude"] <= -83.735390+0.003)]

#2.
trips, counts = np.unique(df["Trip"], return_counts=True)
trips = trips[counts>100]
df = df[df['Trip'].isin(trips)]

#Affichage du dataframe
df

# %% [markdown]
# ### Detail sur l'intervalle de temps entre chaque récupération de données

# %%
#time est une liste qui contiendra le temps passé entre deux récupérations de données pour tous les trips
time = np.array([])

#Parcourt sur les trips car le temps de départ pour les trips est différent
for t in trips:
    #Récupération des données de temps du GPS pour le trip t
    tr = df.loc[df["Trip"]==t, "GpsTime"]
    #Rajout des différences de temps entre les lignes de données dans la liste time
    time = np.concatenate((time,[np.abs(tr.iloc[i+1] - tr.iloc[i]) for i in range(tr.shape[0]-1)]))

#Récupération des valeurs uniques et de leur effectif
values_time, counts = np.unique(time, return_counts=True)
print(f"Les différentes valeurs du temps passé entre deux points :\n{values_time}\n")
print(f"Et leur effectif :\n{counts}")

# %% [markdown]
# On choisit d'ignorer les outliers (ceux qui apparaissent une seule fois) :

# %%
values_time = values_time[counts>1]
counts = counts[counts>1]
plt.title("Distribution des écarts de temps entre deux récupérations")
plt.hist(values_time, weights=counts)

# %% [markdown]
# Peut-on en déduire que l'échantillonnage des points est fixe (avec un écart de 200 millisecondes) ?
# %% [markdown]
# ### Exploration des données
# %% [markdown]
# **Quelques statistiques de base sur certaines colonnes ...**

# %%
colonnes = ['Time','GpsTime','GpsHeading', 'GpsSpeed','Latitude', 'Longitude', 'Altitude','Pdop']

df[colonnes].describe()

# %% [markdown]
# **Visualisation des colonnes Longitude et Latitude :**

# %%
sns.pairplot(df[["Longitude", "Latitude"]], height=3)

# %% [markdown]
# **Distribution de la vitesse des GPS :**

# %%
fig, ax =plt.subplots(1,2, figsize=(12,5))
ax[0].set_title('Distribution de la vitesse des GPS')
sns.histplot(df["GpsSpeed"], kde=True, ax=ax[0])
ax[1].set_title('Distribution de la vitesse >0.05 des GPS')
sns.histplot(df.loc[df["GpsSpeed"]>0.05, "GpsSpeed"], kde=True, ax=ax[1])
plt.show()

# %% [markdown]
# **Quelques fonctions d'affectation et de visualisations**

# %%
#librarie pour dessiner un rectangle
import matplotlib.patches as patches

#Fonction pour affecter des points du dataframe à une case sur un plan
def affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y):
    """ DataFrame * float * float * flot * float -> Series(int) * Series(int)
        Retourne l'affectation des points du DataFrame en deux Series,
        le premier stock les indices x et le second les indices y.
    """
    x = ((df["Latitude"] - latitude_min)/ecart_x).apply(math.floor)
    y = ((df["Longitude"] - longitude_min)/ecart_y).apply(math.floor)
    
    return x,y

#Permet de sélectionner tous les points appartenant à une case
def trouve_data_case(df, pos, latitude_min, longitude_min, ecart_x, ecart_y):
    """DataFrame * (int,int) * float * float * flot * float -> DataFrame
        Retourne un DataFrame contenant toutes les lignes se situant dans la case pos.
    """
    x, y = affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y)
    i, j = pos
    return df[(x==i) & (y==j)]

#Dessine un rectangle sur les cases où se trouvent les données sélectionnées
def dessine_rect(df, ax, x_splits, y_splits, latitude_min, longitude_min, ecart_x, ecart_y):
    """ DataFrame * AxesSubplot * float * float * flot * float -> None
        Encadre les cases correspondant aux points du df.
    """
    #calcul des indices x et y des cases pour chaque ligne de df
    x, y = affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y)
    #pour chaque ligne de df, on encadre en noir sa case
    for i in range(len(x)):
        x_i = x.iloc[i]
        y_i = y.iloc[i]
        ax.add_patch(patches.Rectangle((x_splits[x_i],y_splits[y_i]), ecart_x, ecart_y, edgecolor = 'black', fill=False))

#Ajout du titre et des noms d'axes
def set_ax_legend(ax, title, xlabel, ylabel):
    """ AxesSubplot * str * str * str -> None
        Ajoute le titre et le nom des axes sur ax.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

#Permet de visualiser le sens de la route
def fleche_sens(df, ax):
    """ DataFrame * AxesSubplot -> None
        Dessine une flèche partant du 1er point et qui est orientée vers le n/2-ème point.
    """
    n = df.shape[0]
    if n > 1:
        x_i = df.iloc[0, tr.columns.get_loc("Latitude")]
        y_i = df.iloc[0, tr.columns.get_loc("Longitude")]
        dx_i = df.iloc[math.floor(n/2), tr.columns.get_loc("Latitude")] - x_i
        dy_i = df.iloc[math.floor(n/2), tr.columns.get_loc("Longitude")] - y_i
        ax.quiver(x_i, y_i, dx_i, dy_i)

# %% [markdown]
# **Calcul des paramètres**

# %%
#Bornes de la longitude
longitude_min = df["Longitude"].min()
longitude_max = df["Longitude"].max()
#Bornes de la latitude
latitude_min = df["Latitude"].min()
latitude_max = df["Latitude"].max()
#bins / nombre d'intervalles
n_interval = 10
#On sépare en n_interval la latitude et la longitude
x_splits = np.linspace(latitude_min,latitude_max, n_interval)
y_splits = np.linspace(longitude_min,longitude_max, n_interval)
#Ecart entre deux intervalles des axes
ecart_x = x_splits[1]-x_splits[0]
ecart_y = y_splits[1]-y_splits[0]

# %% [markdown]
# **Calcul de l'effectif et de la vitesse moyenne de chaque case:**

# %%
#Calcul de l'effectif et de la vitesse moyenne sur chaque case
effectif_cases = np.zeros((n_interval,n_interval))
vitesse_cases = np.zeros((n_interval,n_interval))
for i in range(n_interval):
    for j in range(n_interval):
        case_df = trouve_data_case(df, (i, j), latitude_min, longitude_min, ecart_x, ecart_y)
        if case_df.shape[0] > 0 :
            effectif_cases[i,j] = case_df.shape[0]
            vitesse_cases[i,j] = case_df["GpsSpeed"].mean()
            
#Création d'une nouvelles colonnes stockant les données sur les portions de route           
sx,sy = affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y)
e = [] #liste effectif moyen pour chaque ligne
v = [] #liste vitesse moyenne pour chaque ligne
for i in range(sx.shape[0]) :
    e.append(effectif_cases[sx.iloc[i],sy.iloc[i]])
    v.append(vitesse_cases[sx.iloc[i],sy.iloc[i]])

df["Effectif_case"] = e    
df["Vitesse_moy_case"] = v

# %% [markdown]
# **Visualisation graphique de la carte :**

# %%
#Tirage aléatoire de lignes du dataframe pour pouvoir se focaliser sur sa case:
#nombre d'exemples à tirer
nb_points = 1
#tirage aléatoire de nb_points lignes
samples = df.sample(nb_points)


fig, ax = plt.subplots(2,2, figsize=(15,12))

#Visualisation (1ème figure):
set_ax_legend(ax[0][0], "Visualisation des effectifs de voiture de la zone étudiée", "Latitude", "Longitude")
p = ax[0][0].scatter(df["Latitude"], df["Longitude"], c=df["Effectif_case"], cmap="RdPu")
cbar = plt.colorbar(p, ax=ax[0][0])
cbar.set_label('Effectif de voiture')

#Visualisation (2ère figure) :
#affichage (latitude,longitude) pour les trips en fonction de la vitesse
set_ax_legend(ax[0][1], 'Visualisation des vitesses de la zone étudiée', "Latitude", "Longitude")
p = ax[0][1].scatter(df["Latitude"], df["Longitude"], c=df["Vitesse_moy_case"], cmap="YlOrRd")
cbar = plt.colorbar(p, ax=ax[0][1])
cbar.set_label('Vitesse moyenne')
    
#affichage grille
for i in range(n_interval):
    x = x_splits[i]
    y = y_splits[i]
    ax[0][0].plot([x,x],[longitude_min, longitude_max], c='grey',  alpha = 0.5)
    ax[0][0].plot([latitude_min,latitude_max],[y,y], c='grey', alpha = 0.5)
    ax[0][1].plot([x,x],[longitude_min, longitude_max], c='grey',  alpha = 0.5)
    ax[0][1].plot([latitude_min,latitude_max],[y,y], c='grey', alpha = 0.5)

#affiche les rectangles correspondant aux données tirées aléatoirement    
dessine_rect(samples, ax[0][0], x_splits, y_splits, latitude_min, longitude_min, ecart_x, ecart_y)
dessine_rect(samples, ax[0][1], x_splits, y_splits, latitude_min, longitude_min, ecart_x, ecart_y)

#si on ne possède qu'une case, alors on fait un zoom dessus
if nb_points == 1:
    
    #Visualisation (3ème figure) :
    sx, sy = affectation_2(samples, latitude_min, longitude_min, ecart_x, ecart_y)
    case_df = trouve_data_case(df, (sx.iloc[0], sy.iloc[0]), latitude_min, longitude_min, ecart_x, ecart_y)
    p = ax[1][0].scatter(case_df["Latitude"], case_df["Longitude"], c=case_df["GpsSpeed"], cmap="YlOrRd")
    cbar = plt.colorbar(p, ax=ax[1][0])
    cbar.set_label('Vitesse')  
    set_ax_legend(ax[1][0], f"Zoom sur la case {(sx.iloc[0],sy.iloc[0])}", "Latitude", "Longitude")
    
    #Affichage du sens de circulation pour la figure 3
    trips_case = np.unique(case_df["Trip"])
    for t in trips_case:
        tr = case_df.loc[case_df["Trip"]==t, ["Latitude","Longitude","GpsHeading"]]
        fleche_sens(tr, ax[1][0])
        
    #Visualisation (4ème figure):
    sns.histplot(case_df["GpsSpeed"],ax=ax[1][1])
    ax[1][1].set_title(f"Distribution de la vitesse sur la case {(sx.iloc[0],sy.iloc[0])}")
          
plt.show()

#Histogramme des valeurs de GpsHeading
plt.figure()
plt.title(f"Histogramme de la colonne GpsHeading de la case {(sx.iloc[0],sy.iloc[0])}")
sns.histplot(case_df["GpsHeading"])
plt.show()

# %% [markdown]
# **Calcul de la norme et de l'angle $\Theta$ des vecteurs vitesse :** 

# %%
case_df = trouve_data_case(df, (sx.iloc[0], sy.iloc[0]), latitude_min, longitude_min, ecart_x, ecart_y)
trips_case = np.unique(case_df["Trip"])

liste_norm_v = []
liste_theta_v = []

for t in trips_case:
    tr = case_df.loc[case_df["Trip"]==t, ["GpsTime","Latitude","Longitude"]]                  
    for i in range(1,tr.shape[0]):
        dif_time = (tr["GpsTime"].iloc[i] - tr["GpsTime"].iloc[i-1])
        v = (tr[["Latitude","Longitude"]].iloc[i] - tr[["Latitude","Longitude"]].iloc[i-1])/dif_time
        norm_v = np.sqrt(v["Latitude"]**2 + v["Longitude"]**2)        
        theta = np.arctan(v["Latitude"]/np.maximum(v["Longitude"], 0.0001))
        
        liste_norm_v.append(norm_v)
        liste_theta_v.append(theta)


# %%
tr


# %%
case_df

# %% [markdown]
# **Histogramme 3D de la norme des vecteurs de vitesse et ses angles $\Theta$ :**

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(liste_norm_v, liste_theta_v, bins=4)

# The start of each bucket.
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

# The width of each bucket.
dx, dy = np.meshgrid(xedges[1:] - xedges[:-1], yedges[1:] - yedges[:-1])

dx = dx.flatten()
dy = dy.flatten()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

ax.set_xlabel('Norm')
ax.set_ylabel('$\Theta$')
ax.set_zlabel('Effectif')

plt.title("Histogramme 3D de la norme des vecteurs de vitesse et ses angles $\Theta$ :")

plt.show()

# %% [markdown]
# # Modele Physique
# %% [markdown]
# ### Explication du modele
# %% [markdown]
# ### Implementation
# %% [markdown]
# ### Test
# %% [markdown]
# ### Base sur les points precedentes

# %%
# chargement des donnees
tr


# %%
def vald_crois(f,datax, n):
    N = len(datax)
    gap = int(N/n)
    err = []
    for i in range(n):
        X_test = datax[i*gap:(i+1)*gap]
        X_train = np.concatenate((datax[0:i*gap],datax[(i+1)*gap:]), axis = 0), np.concatenate((datay[0:i*gap],datay[(i+1)*gap:]), axis = 0)
        f.fit(X_train)
        err.append(1 - f.score(X_test, Y_test))
    return err


# %%
datax = []
tr_arr = tr.to_numpy()
for i in range(len(tr)-2):
    datax.append([tr_arr[i],tr_arr[i+1],tr_arr[i+2]])
    i+=3

datax = np.array(datax)

datax


# %%
model_phy = mp.modele_physique()

def moindre_c(X_predit, X_test):
    return ((X_predit-X_test)**2).sum()


# %%
model_phy.score(moindre_c, datax)

# %% [markdown]
# ### Base sur l'attribut GpsHeading et GpsSpeed

# %%
df = case_df[['Latitude','Longitude','GpsHeading','GpsSpeed','GpsTime']]


# %%
df


# %%
arr = df.to_numpy()
arr[:,:-1]


# %%
model_phy.predictFromInstantSpeed(arr[:,:-1], 0.2)


# %%
def toRadians(v):
    return v*np.pi / 180

def toDegrees(v):
    return v*180/np.pi   

def toNordBasedHeading(GpsHeading):
    return 90 - GpsHeading

def predictFromInstantSpeed(x_test, alpha):
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


# %%
predictFromInstantSpeed(arr[:,:-1], 0.2)


# %%
sns.histplot(df["GpsHeading"])


