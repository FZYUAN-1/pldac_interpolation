from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import dataSource as ds
import dataVisualization as dv
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
#  Class Traitement
# =============================================================================

class Traitement:
    """
        La classe Traitement permet de construire les données X,y 
        d'apprentissage et de test.
    """
    
    def __init__(self, df, l_attrs_x, labels, freq_train=1000, freq_test=400, preprocessor=None):
        #DataFrame
        self.df = df
        #features
        self.l_attrs = l_attrs_x
        #targets
        self.labels = labels
        #preprocessor
        self.preprocessor = preprocessor
        #fréquences
        self.freq_train = freq_train
        self.freq_test = freq_test

        #Données d'apprentissage/test pour chaque modèle
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        
    # Fonction de construction des données de train/test
    def set_data_train_test(self, train_size=0.8):
        X_train, X_test, y_train, y_test = ds.create_data_xy(self.df, train_size, self.freq_train, self.freq_test)
        
        #On vide les anciennes données s'il y en a
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        for attrs in self.l_attrs:
            self.l_Xtrain.append(X_train[attrs])
            self.l_Xtest.append(X_test[attrs])
            self.l_Ytrain.append(y_train[self.labels])
            self.l_Ytest.append(y_test[self.labels])


# =============================================================================
#  Class Evaluation 
# =============================================================================

class Evaluation :
    """
        La classe Evaluation permet d'entrainer des modèles à partir de la
        classe Traitement et d'en afficher des résultats.
    """
    
    def __init__(self, models, traitement):
        self.models = models
        self.traitement = traitement
        self.preprocessor = self.traitement.preprocessor

        self.l_Xtrain = self.traitement.l_Xtrain
        self.l_Ytrain = self.traitement.l_Ytrain
        self.l_Xtest = self.traitement.l_Xtest
        self.l_Ytest = self.traitement.l_Ytest
        self.labels = self.traitement.labels
        
        #Ajout du preprocessor à la pipeline s'il y en a un
        if self.preprocessor is not None :
            self.models_pip = [make_pipeline(self.preprocessor[i], self.models[i]) for i in range(len(self.models))]
        else: 
            self.models_pip = self.models
    
        
    
    def fit(self):
        """
            Fonction qui entraine tous nos modèles.
        """
        for i in range(len(self.models)):
            self.models_pip[i].fit(self.l_Xtrain[i], self.l_Ytrain[i])
            
    def score(self):
        """
            Fonction retournant une liste de scores sur les données de test
            pour chaque modèle.
        """
        return [self.models_pip[i].score(self.l_Xtest[i], self.l_Ytest[i]) for i in range(len(self.models))]
    
    def predict(self, X):
        """
            Fonction retournant une liste de prédiction sur X pour chaque
            modèle.
        """
        return [self.models_pip[i].predict(X[i]) for i in range(len(self.models))]
  
    def getCoef(self):
        """
            Fonction retournant les paramètres appris pour chaque modèle.
        """
        return [self.models[i].coef_ for i in range(len(self.models))]
    
    def calculMse(self):
        ypred = self.predict(self.l_Xtest)
        return [mean_squared_error(self.l_Ytest[i],ypred[i]) for i in range(len(self.models))]
    
    
    # ------------------------- Fonctions d'affichage -------------------------
    
    def afficher_score(self):
        """
            Fonction affichant les scores pour chaque modèle.
        """
        scores = self.score()
        for i in range(len(self.models)):
            print(f"Score obtenu pour le modèle {i : <10} : {scores[i]}")
            
    def afficher_coef(self):
        """
            Fonction affichant les coefficients pour chaque modèle.
        """
        coefs = self.getCoef()
        for i in range(len(self.models)):
            print(f"Coefficients obtenu pour le modèle {i : <10} : {coefs[i]}")
            
    def afficher_mse(self):
        ypred = self.predict(self.l_Xtest)
        for i in range(len(self.models)):
            print(f"MSE obtenue pour le modèle  {i : <10} : {mean_squared_error(self.l_Ytest[i],ypred[i])}")
        
    def afficher_resultats(self):
        """
            Fonction appelant les autres fonctions d'affichage.
        """
        self.afficher_score()
        print()
        self.afficher_mse()
        print()
        #self.afficher_coef()
        
    # ----------------------------- Fonctions MSE -----------------------------
    
    def tabMSEFreq(self, liste_freq, train_size=0.8):
        tab_mse = []
        models = [deepcopy(m) for m in self.models]
        
        for freq in liste_freq:
            traitement  = Traitement(self.traitement.df, self.traitement.l_attrs, self.traitement.labels,
                                     freq, freq, self.traitement.preprocessor)
            traitement.set_data_train_test(train_size)
            
            evaluateur = Evaluation(models,traitement)
            evaluateur.fit()
            
            tab_mse.append(evaluateur.calculMse())
        
        tab_mse = np.array(tab_mse)
            
        #Affichage des erreurs MSE des modèles en fonction de la fréquence    
        plt.figure(figsize=(7,5))
        plt.title("Erreur MSE en fonction de la fréquence")
        for i in range(len(models)):
            plt.plot(tab_mse[:,i], label=type(models[i]).__name__)
            
        plt.xticks(np.arange(len(liste_freq)), np.array(liste_freq))
        plt.xlabel("Fréquences")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()
            
        return tab_mse
    
    def matMSECase(self, freq_train, freq_test, lat_min, long_min, e_x, e_y, min_datapts=20, train_size=0.8, n_interval=10):
        models = [deepcopy(m) for m in self.models]
        # liste matrices erreurs des cases
        l_mat_err= [np.zeros((n_interval, n_interval)) for i in range(len(models))]
        
        # parcours de toutes les cases
        for i in range(n_interval):
            for j in range(n_interval):
                # récupération des données de la case
                case_df=ds.trouve_data_case(self.traitement.df, (i, j), lat_min, long_min, e_x, e_y)
    
                #On prend les Trips qui ont au moins $min_datapoints$ points
                #c'est pas au moins 2 points car tu splits en train et en test, ca aura moins d'un point 
                ctrips, ccounts = np.unique(case_df["Trip"], return_counts=True)
                ctrips = ctrips[ccounts>min_datapts]
                case_df = case_df[case_df['Trip'].isin(ctrips)]
    
                #Cases qui ont au moins 2 trips
                if len(pd.unique(case_df["Trip"])) > 1 :
                    traitement = Traitement(case_df, self.traitement.l_attrs, self.traitement.labels, 
                                            freq_train, freq_test, self.traitement.preprocessor)
                    traitement.set_data_train_test(train_size)
    
                    l_ypred = self.predict(traitement.l_Xtest)
    
                    for mi in range(len(models)):               
                        l_mat_err[mi][n_interval-1-i, j] = mean_squared_error(traitement.l_Ytest[mi],l_ypred[mi])
    
    
        for m in range(len(l_mat_err)):
            fig, ax = plt.subplots(1,2, figsize=(15,5))
            ax[0].set_title(f"Erreur MSE par case : {type(models[m]).__name__}")
            sns.heatmap(l_mat_err[m], linewidths=.5, cmap="YlGnBu", yticklabels=np.arange(n_interval-1, -1, -1), ax=ax[0])
            ax[1].set_title("Valeurs de la matrice en histogramme")
            val = l_mat_err[m].ravel()[l_mat_err[m].ravel() != 0]
            sns.histplot(val, ax=ax[1])
        plt.show()
        