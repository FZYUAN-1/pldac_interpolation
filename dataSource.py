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
	df = pd.read_csv("DataGpsDas.csv", nrows=1000000)
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
	x_splits = np.linspace(latitude_min,latitude_max, n_interval)
	y_splits = np.linspace(longitude_min,longitude_max, n_interval)
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
	
	return x,y


#Permet de sélectionner tous les points appartenant à une case
def trouve_data_case(df, pos, latitude_min, longitude_min, ecart_x, ecart_y, step=1):
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
		
	return e, v


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
def create_data_xy(df, attrs_x, label, step_train, step_test):
    """ Renvoie les DataFrames X et Y, en fonction des paramètres.

        @params :
            df      : DataFrame : Données à traiter
            attrs_x : list(str) : Liste des attributs dans les données d'apprentissage
            label   : str       : Attributs du label
            step    : int       : Le pas à prendre pour la sélection des lignes
            
        @return :
            DataFrame, DataFrame           
    """
    data_x = []
    data_y = []
    '''
    trips = np.unique(df["Trip"])

    for t in range(len(trips)):
        trip_df = df[df['Trip'] == trips[t]]
        if t == 0:
            data_x = echantillon(trip_df[:-step], step)[attrs_x]
            data_y = echantillon(trip_df[step:], step)[label]

        else :
            x = echantillon(trip_df[:-step], step)[attrs_x]
            y = echantillon(trip_df[step:], step)[label]
            data_x = pd.concat([data_x,x])
            data_y = pd.concat([data_y,y])
    '''
    groups = df.groupby('Trip')
    for group in groups:
        trip_i = group[1][attrs_x]
        data_x.append(echantillon(trip_i[:-step], step)[attrs_x].to_numpy())
        data_y.append(echantillon(trip_i[step:], step)[label].to_numpy())
        
    return data_x, data_y

def train_test_split(df,attrs_x, labels,freq_train,freq_test):
    '''

    parameters:
    -----------
    datax : [[attrs_x]*N]
    datay : [[labels]*N]
    '''
        
    X_train = []
    X_test = []
    y_train = [] 
    y_test = []
    
    
    step_train = freq_train//200
    step_test = freq_test//200

    groups = df.groupby('Trip')
    for group in groups:
        trip_i = group[1]
        
        datax = trip_i[attrs_x].to_numpy()
        datay = trip_i[labels].to_numpy()

        #print(datax.shape)
        tmp1 = datax[:-step_train:step_train]
        tmp2 = datay[step_train::step_train]
        X_train.append(tmp1)
        y_train.append(tmp2)

        #print(X_train)
        #print(y_train)
        '''
        l = []
        for point in datax[:-step_test:step_test]:
            
            l.append(point)
        X_test.append(l)

 
        l = []
        for point in datay[step_test::step_test]:
            if point in tmp2:
                continue
            else:
                l.append(point)
        y_test.append(l)
        '''
        X_test.append(datax[:-step_test:step_test])
        y_test.append(datay[step_test::step_test])

    return X_train, X_test, y_train, y_test

'''
        l1 = []
        l2 = []
        for j in range(len(idx_datax)-1):
            l1.append([ datax[idx_datax[j]+1:idx_datax[j+1]:step_test] ])
            l2.append([ datay[idx_datay[j]+1:idx_datay[j+1]:step_test] ])
        
        l1.append(datax[idx_datax[len(idx_datax)-1]+1::step_test])
        l2.append(datay[idx_datay[len(idx_datay)-1]+1::step_test])

        X_test.append(l1)
        y_test.append(l2)
'''

        # X_test est une liste d'intervalles et chaque intervalle contient les points de test que nous voulons faire l'interpolation avec
        #X_test.append([ datax[i][idx_train[j]+1:idx_train[j+1]:step_test] for j in range(len(idx_train)-1)])
        # Pareil pour y_test
        #X_test.append([ datay[i][idx_train[j]+1:idx_train[j+1]:step_test] for j in range(len(idx_train)-1)])


'''
df = importData()
pos = (4,4)
latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y = calcul_param(df)
case = trouve_data_case(df,pos,latitude_min, longitude_min, ecart_x, ecart_y)

#datax,datay = create_data_xy(case,['Latitude','Longitude','GpsTime'],['Latitude','Longitude','GpsTime'])

X_train, X_test, y_train, y_test = train_test_split(case,['Latitude','Longitude','GpsTime'],['Latitude','Longitude','GpsTime'],1000,400)

#print(datax[0],datax[1],datay[0],datay[1])
print(X_train[0],X_test[0],y_train[0],y_test[0])
'''


