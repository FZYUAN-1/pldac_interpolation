import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import dataSource as ds

# =============================================================================
# Fonctions de visualisations
# =============================================================================

#Affiche un histogramme pour un attribut du DataFrame
def afficher_histogramme_df(df,attr):
	""" DataFrame * str -> None
	
		Affiche un histogramme sur un attribut du DataFrame df.
	"""
	plt.figure()
	sns.histplot(df[attr])
	plt.show()
	
	
#Dessine un rectangle sur les cases où se trouvent les données sélectionnées
def dessine_rect(ax, pos, x_splits, y_splits, ecart_x, ecart_y):
	""" AxesSubplot * (int,int) * float * float * flot * float -> None
	
		Encadre la case correspondant à la position.
	"""
	x, y = pos
	ax.add_patch(patches.Rectangle((x_splits[x],y_splits[y]), ecart_x, ecart_y, edgecolor = 'black', fill=False))


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
		x_i = df.iloc[0, df.columns.get_loc("Latitude")]
		y_i = df.iloc[0, df.columns.get_loc("Longitude")]
		dx_i = df.iloc[math.floor(n/2), df.columns.get_loc("Latitude")] - x_i
		dy_i = df.iloc[math.floor(n/2), df.columns.get_loc("Longitude")] - y_i
		ax.quiver(x_i, y_i, dx_i, dy_i)
		

#Visualisation graphique de la carte
def affiche_carte(df, pos, latitude_min, latitude_max, longitude_min, longitude_max, ecart_x, ecart_y, n_interval=10) :
	""" DataFrame * (int,int) * float * float * float * float * float * float * int -> None
	
		Affiche un aperçu de la carte et d'une case donnée.
	"""
	#Préparation des données
	
	#On sépare en n_interval la latitude et la longitude
	x_splits = np.linspace(latitude_min,latitude_max, n_interval)
	y_splits = np.linspace(longitude_min,longitude_max, n_interval)
	
	#Ajout des colonnes de l'effectif et de la vitesse moyenne d'une case
	e, v = ds.calcul_eff_vit_moy(df, latitude_min, longitude_min, ecart_x, ecart_y)
	df["Effectif_case"] = e    
	df["Vitesse_moy_case"] = v
	
	
	#Affichage 
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
	
	dessine_rect(ax[0][0], pos, x_splits, y_splits, ecart_x, ecart_y)
	dessine_rect(ax[0][1], pos, x_splits, y_splits, ecart_x, ecart_y)
	
	#Visualisation (3ème figure) :
	sx, sy = pos
	case_df = ds.trouve_data_case(df, (sx, sy), latitude_min, longitude_min, ecart_x, ecart_y)
	p = ax[1][0].scatter(case_df["Latitude"], case_df["Longitude"], c=case_df["GpsSpeed"], cmap="YlOrRd")
	cbar = plt.colorbar(p, ax=ax[1][0])
	cbar.set_label('Vitesse')  
	set_ax_legend(ax[1][0], f"Zoom sur la case {(sx,sy)}", "Latitude", "Longitude")
	
	#Affichage du sens de circulation pour la figure 3
	trips_case = np.unique(case_df["Trip"])
	for t in trips_case:
		tr = case_df.loc[case_df["Trip"]==t, ["Latitude","Longitude","GpsHeading"]]
		fleche_sens(tr, ax[1][0])
		
	#Visualisation (4ème figure):
	sns.histplot(case_df["GpsSpeed"],ax=ax[1][1])
	ax[1][1].set_title(f"Distribution de la vitesse sur la case {(sx,sy)}")
			  
	plt.show()
		

#Histogramme 3D de la norme des vecteurs de vitesse et ses angles Θ	
def afficher_hist_norm_vit(df, pos, latitude_min, longitude_min, ecart_x, ecart_y):
    """ DataFrame * (int,int) * float * float * float * float -> None
	
		Affiche un histogramme 3d par rapport aux normes et anlges des vecteurs vitesse
        d'une case donnée.
	"""
    #Préparation des données
    liste_norm_v, liste_theta_v = ds.calcul_norm_theta(df, pos, latitude_min, longitude_min, ecart_x, ecart_y)
    
    #Affichage 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Histogramme 3D de la norme des vecteurs de vitesse et ses angles $\Theta$ :")
    ax.set_xlabel('Norm')
    ax.set_ylabel('$\Theta$')
    ax.set_zlabel('Effectif')
    
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
    
    plt.show()


#Affichage de la matrice en heatmap
def afficher_mat(mat, title="Matrice en heatmap", n_interval=10):
    """ Matrice(float) * int -> None
    
        Affiche un heatmap de la matrice.
    """
    plt.figure()
    plt.title(title)
    sns.heatmap(mat, linewidths=.5, cmap="YlGnBu",
            yticklabels=np.arange(n_interval-1, -1, -1))
    plt.show()
    
    
#Histogramme des valeurs de la matrice
def afficher_mat_hist(mat):
    """ Matrice(float) -> None
    
        Affiche les valeurs de la matrice sous histogramme.
    """
    #On commence par sélectionner que les cases contenant une valeur
    val = mat.ravel()[mat.ravel() != 0]

    #Affichage des valeurs sous histogramme
    plt.figure()
    plt.title("Affichage des valeurs de la matrice sous histogramme")
    sns.histplot(val)
    plt.show()		
		