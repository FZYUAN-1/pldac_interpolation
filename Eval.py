from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from math import ceil, floor
import matplotlib.pyplot as plt


# =============================================================================
#  Class Traitement
# =============================================================================

class Traitement:
    """
        La classe Traitement permet de construire les données X,y 
        d'apprentissage et de test.
    """
    
    def __init__(self, df, attrs_x, labels, preprocessor=None):
        #DataFrame
        self.df = df
        #features
        self.attrs = attrs_x
"""        self.l_attrs = attrs_x  # attrs_x étant une liste des listes d'attributs"""
        #targets
        self.labels = labels
        #preprocessor
        self.preprocessor = preprocessor
        
        #liste des données d'apprentissage/test pour chaque modèle
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        
    # Fonction de construction des données de train/test en fonction de la méthode 
    # passée en argument
    
    def set_data_train_test(self, func, step=1, test_size=0.2, random_state=0):
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        for label in self.labels:
            data_x, data_y = func(self.df, self.attrs, label, step)
            X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=random_state)
            self.l_Xtrain.append(X_train)
            self.l_Xtest.append(X_test)
            self.l_Ytrain.append(y_train)
            self.l_Ytest.append(y_test)

"""     for attrs in self.l_attrs   
            data_x, data_y = func(self.df, attrs, labels, step)
            X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size, random_state=random_state)
            self.l_Xtrain.append(X_train)
            self.l_Xtest.append(X_test)
            self.l_Ytrain.append(y_train)
            self.l_Ytest.append(y_test)

    # self.l_attrs = [[attrs for model1], [attrs for model2]...]
    # l_Xtrain = [[Xtrain for attrs1], [for attrs2]...] """
    
    # Getters
    def getPreprocessor(self):
        return self.preprocessor
    
    def getLXtrain(self):
        return self.l_Xtrain
    
    def getLXtest(self):
        return self.l_Xtest
    
    def getLYtrain(self):
        return self.l_Ytrain
    
    def getLYtest(self):
        return self.l_Ytest
    
    def getLabels(self):
        return self.labels
    

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
        self.preprocessor = traitement.getPreprocessor()
        self.l_Xtrain = traitement.getLXtrain()
        self.l_Ytrain = traitement.getLYtrain()
        self.l_Xtest = traitement.getLXtest()
        self.l_Ytest = traitement.getLYtest()
        self.labels = traitement.getLabels()
        
        #Ajout du preprocessor à la pipeline s'il y en a un
        if self.preprocessor is not None :
            self.models_pip = [make_pipeline(self.preprocessor, model) for model in self.models]
        else: 
            self.models_pip = self.models
    

    def fit(self):
        """
            Fonction quit entraine tous nos modèles.
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
        return [self.models_pip[i].predict(X) for i in range(len(self.models))]
  
    def getCoef(self):
        """
            Fonction retournant les paramètres appris pour chaque modèle.
        """
        return [self.models[i].coef_ for i in range(len(self.models))]
    
    

    # Fonctions d'affichage
    
    def afficher_score(self):
        """
            Fonction affichant les scores pour chaque modèle.
        """
        scores = self.score()
        for i in range(len(self.models)):
            print(f"Score obtenu pour le label {self.labels[i] : <10} : {scores[i]}")
            
    def afficher_coef(self):
        """
            Fonction affichant les coefficients pour chaque modèle.
        """
        coefs = self.getCoef()
        for i in range(len(self.models)):
            print(f"Coefficients obtenu pour le label {self.labels[i] : <10} : {coefs[i]}")
            
    def afficher_mse(self):
        l_ypred = self.predict(self.l_Xtest[0])
        for i in range(len(self.models)):
            print(f"MSE obtenue pour le label {self.labels[i] : <10} : {mean_squared_error(self.l_Ytest[i],l_ypred[i])}")
    
    def afficher_pred(self):
        """
            Fonction affichant les prédictions sous forme graphique.
        """
        nb_lignes = ceil(len(self.labels)/2)
        l_ypred = self.predict(self.l_Xtest[0])
        
        fig, axs = plt.subplots(nb_lignes,2, figsize = (15,nb_lignes*5))

        if nb_lignes > 1:
            for i in range(len(self.l_Ytest)):
                axs[floor(i/2)][i%2].set_title(self.labels[i])
                axs[floor(i/2)][i%2].scatter(self.l_Xtest[i][self.labels[i]],self.l_Ytest[i])
                axs[floor(i/2)][i%2].plot(self.l_Xtest[i][self.labels[i]],l_ypred[i], c="r")

        elif nb_lignes == 1:
            for i in range(len(self.l_Ytest)):
                axs[i%2].set_title(self.labels[i])
                axs[i%2].scatter(self.l_Xtest[i][self.labels[i]],self.l_Ytest[i])
                axs[i%2].plot(self.l_Xtest[i][self.labels[i]],l_ypred[i], c="r")
        plt.show()
        
               
        
    def afficher_resultats(self):
        """
            Fonction appelant les autres fonctions d'affichage.
        """
        self.afficher_score()
        print()
        self.afficher_mse()
        print()
        self.afficher_coef()
        print()
        self.afficher_pred()