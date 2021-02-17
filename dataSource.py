import math
import numpy as np
import pandas as pd

# =============================================================================
# Fonctions récupération/traitements des données
# =============================================================================

#Importation des données
def importData():
	"""
	Importation des données du fichier DataGpsDas.csv

	Returns
	-------
	df : DataFrame
		Les données du fichier csv sous forme de Dataframe (Pandas)

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
	"""
	Calcul des paramètres bornes longitudes, latitudes et l'écarts entre les intervalles.

	Parameters
	----------
	df : DataFrame
		Le Dataset sur lequel on calcule les paramètres
	n_interval : int
		Le nombre d'intervalles.

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











