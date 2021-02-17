import math
import numpy as np
import pandas as pd

#Calcul des paramètres
def calcul_param(df, n_interval=10):
	"""
	Calcul des paramètres bornes longitudes, latitudes et l'écarts entre les intervalles

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


