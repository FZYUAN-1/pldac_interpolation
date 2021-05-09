import math
import numpy as np
import pandas as pd

# =============================================================================
# Fonctions récupérations de données
# =============================================================================

#Importation des données
def importData():
    """ None -> DataFrame
    
    Importation des données du fichier DataGpsDas.csv en respectant certaines contraintes.
    """
    df = pd.read_csv("../DataGpsDas.csv", nrows=1000000)
    df = df[(df["Latitude"] >= 42.282970-0.003) & (df["Latitude"] <= 42.282970+0.003) 
            & (df["Longitude"] >= -83.735390-0.003) & (df["Longitude"] <= -83.735390+0.003)]
    trips, counts = np.unique(df["Trip"], return_counts=True)
    trips = trips[counts>100]
    df = df[df['Trip'].isin(trips)]
    return df


#Calcul des paramètres
def calcul_param(df, n_interval=10):
    """ DataFrame * int -> float * float * float * float * float * float 
    
    Calcul des paramètres bornes longitudes, latitudes et l'écarts entre les intervalles.

    Returns
    -------
    latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y 

    """
    #Bornes de la longitude
    longitude_min = df["Longitude"].min()
    longitude_max = df["Longitude"].max()
    #Bornes de la latitude
    latitude_min = df["Latitude"].min()
    latitude_max = df["Latitude"].max()
    #bins / nombre d'intervalles
    #On sépare en n_interval la latitude et la longitude
    x_splits = np.linspace(latitude_min,latitude_max, n_interval+1)
    y_splits = np.linspace(longitude_min,longitude_max, n_interval+1)

    #Ecart entre deux intervalles des axes
    ecart_x = x_splits[1]-x_splits[0]
    ecart_y = y_splits[1]-y_splits[0]
    
    return latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y


#Fonction pour affecter des points du dataframe à une case sur un plan
def affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y):
    """ DataFrame * float * float * float * float -> Series(int) * Series(int)
    
        Retourne l'affectation des points du DataFrame en deux Series,
        le premier stock les indices x et le second les indices y.
    """  
    x = ((df["Latitude"] - latitude_min)/ecart_x).apply(math.floor)
    y = ((df["Longitude"] - longitude_min)/ecart_y).apply(math.floor)
    #x = ((df["Latitude"] - latitude_min)/ecart_x).apply(arrondi)
    #y = ((df["Longitude"] - longitude_min)/ecart_y).apply(arrondi)
    
    return x,y

def arrondi(x):
    eps = 1e-8
    res = x
    if np.abs(x-math.ceil(x)) < eps:
       res = math.ceil(x)
    else:
       res = math.floor(x)
    return res
        
        
#Permet de sélectionner tous les points appartenant à une case
def trouve_data_case(df, pos, latitude_min, longitude_min, ecart_x, ecart_y):
    """ DataFrame * (int,int) * float * float * float * float -> DataFrame
    
        Retourne un DataFrame contenant toutes les lignes se situant dans la case pos.
    """
    x, y = affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y)
    i, j = pos
    return df[(x==i) & (y==j)]


#Calcul de l'effectif et de la vitesse moyenne de la case pour chaque point du DataFrame
def calcul_eff_vit_moy(df,  latitude_min, longitude_min, ecart_x, ecart_y, n_interval=10):
    """ DataFrame * float * float * float * float * int -> list(int) * list(float)
    
        Retourne l'effectif et la vitesse moyenne de la case du point pour toutes les 
        lignes du df.
    """
    
    effectif_cases = np.zeros((n_interval,n_interval))
    vitesse_cases = np.zeros((n_interval,n_interval))
    vitesse_var = np.zeros((n_interval,n_interval))
    for i in range(n_interval):
        for j in range(n_interval):
            case_df = trouve_data_case(df, (i, j), latitude_min, longitude_min, ecart_x, ecart_y)
            if case_df.shape[0] > 0 :
                effectif_cases[i,j] = case_df.shape[0]
                vitesse_cases[i,j] = case_df["GpsSpeed"].mean()
                vitesse_var[i,j] = case_df["GpsSpeed"].var()
                
    #Création d'une nouvelles colonnes stockant les données sur les portions de route           
    sx,sy = affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y)

    sx.replace(n_interval, n_interval-1, inplace=True)
    sy.replace(n_interval, n_interval-1, inplace=True)
    
    e = [] #liste effectif moyen pour chaque ligne
    v = [] #liste vitesse moyenne pour chaque ligne
    v2 = [] #liste varaince vitesse pour chaque ligne
    
    for i in range(sx.shape[0]) :
        e.append(effectif_cases[sx.iloc[i],sy.iloc[i]])
        v.append(vitesse_cases[sx.iloc[i],sy.iloc[i]])
        v2.append(vitesse_var[sx.iloc[i],sy.iloc[i]])
        
    return e, v, v2


# Calcul de la norme et de l'angle  Θ  des vecteurs vitesse par rapport au point précédent
def calcul_norm_theta(df, pos, latitude_min, longitude_min, ecart_x, ecart_y):
    """ DataFrame * (int,int) * float * float * float * float -> list(float) * list(float)

        Retourne les listes de normes et d'angles du vecteur vitesse de la case pos par 
        rapport au point précédent.
    """
    case_df = trouve_data_case(df, pos, latitude_min, longitude_min, ecart_x, ecart_y)
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
            
    return liste_norm_v, liste_theta_v


# =============================================================================
# Fonctions traitements de données
# =============================================================================


def echantillon(df, step=1):
    """ DataFrame * int -> DataFrame
        
        Sélectionne une ligne sur 'step' dans le DataFrame.
    """
    ind = np.arange(0,df.shape[0],step)
    return df.iloc[ind]

#Création des données d'apprentissage pour la prédiction du prochain point
def create_data_xy(df, train_size, freq_train, freq_test):
    """ Renvoie les DataFrames X et Y, en fonction des paramètres.

        @params :
            df      : DataFrame : Données à traiter
            train_size : float  : Pourcentage de données train
            freq_train : int    : Freq à prendre sur train
            freq_test  : int    : Freq à prendre sur test
            
        @return :
            DataFrame, DataFrame   
    """
    #Fréquences des données
    # np.random.seed(0)
    
    step_train = freq_train//200
    step_test = freq_test//200
    #Sélection des numéros de Trip en train et en test
    trips = pd.unique(df["Trip"])
    # melange = np.arange(len(trips))
    # np.random.shuffle(melange)
    # train_trips = trips[melange[:int(len(melange)*train_size)]]
    # test_trips = trips[melange[int(len(melange)*train_size):]]
    train_trips = trips[:int(len(trips)*train_size)]
    test_trips = trips[int(len(trips)*train_size):]
    
    #Création des DataFrame train/test
    X_train = None
    X_test = None
    y_train = None
    y_test = None


    #Construction des données d'apprentissage
    for t in range(len(train_trips)):
        train_df = df[df['Trip'] == train_trips[t]]
        if t == 0:
            X_train = echantillon(train_df[:-step_train], step_train)
            y_train = echantillon(train_df[step_train:], step_train)
        else :
            xtrain = echantillon(train_df[:-step_train], step_train)
            ytrain = echantillon(train_df[step_train:], step_train)
            X_train = pd.concat([X_train,xtrain])
            y_train = pd.concat([y_train,ytrain])
            
    #Construction des données de test      
    for t in range(len(test_trips)):
        test_df = df[df['Trip'] == test_trips[t]]
        
        if t == 0:
            X_test = echantillon(test_df[:-step_test], step_test)
            y_test = echantillon(test_df[step_test:], step_test)
        else :
            xtest = echantillon(test_df[:-step_test], step_test)
            ytest = echantillon(test_df[step_test:], step_test)
            X_test = pd.concat([X_test,xtest])
            y_test = pd.concat([y_test, ytest])
        
    return X_train, X_test, y_train, y_test





def create_data_xy2(df, train_size, freq_train, freq_test):

    step_train = freq_train//200
    step_test = freq_test//200
    #Sélection des numéros de Trip en train et en test
    trips = pd.unique(df["Trip"])
    train_trips = trips[:int(len(trips)*train_size)]
    test_trips = trips[int(len(trips)*train_size):]
    
    X_train1 = None
    X_train2 = None
    y_train = None

    for t in range(len(train_trips)):
        train_df = df[df['Trip'] == train_trips[t]]
        if t == 0:
            X_train1 = echantillon(train_df[step_train*0:-step_train*2], step_train*3)
            X_train2 = echantillon(train_df[step_train:-step_train*1], step_train*3)
            y_train  = echantillon(train_df[step_train*2:], step_train*3)
        else :
            tmp1 = echantillon(train_df[step_train*0:-step_train*2], step_train*3)
            tmp2 = echantillon(train_df[step_train:-step_train*1], step_train*3)
            tmp3  = echantillon(train_df[step_train*2:], step_train*3)

            X_train1 = pd.concat([X_train1,tmp1])
            X_train2 = pd.concat([X_train2,tmp2])
            y_train = pd.concat([y_train,tmp3])
    
    return X_train1, X_train2, y_train