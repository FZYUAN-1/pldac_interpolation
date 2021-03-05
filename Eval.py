from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from dataSource import create_data_xy, train_test_split

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

        self.freq_train = freq_train
        self.freq_test = freq_test

        #Données d'apprentissage/test pour chaque modèle
        self.l_Xtrain = []
        self.l_Ytrain = []
        self.l_Xtest = []
        self.l_Ytest = []
        
        
    # Fonction de construction des données de train/test en fonction de la méthode 
    # passée en argument
    
    def set_data_train_test(self):
        for attrs in self.l_attrs:

            X_train, X_test, y_train, y_test = train_test_split(self.df, attrs, self.labels, self.freq_train, self.freq_test)

            # ========================================================================
            # X_train, X_test, y_train, y_test are consisted of every trip
            # X_train[0] -> attrs for the first trip
            # ========================================================================

            self.l_Xtrain.append(X_train)
            self.l_Xtest.append(X_test)
            self.l_Ytrain.append(y_train)
            self.l_Ytest.append(y_test)


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
        self.preprocessor = traitement.preprocessor

        self.l_Xtrain = traitement.l_Xtrain
        self.l_Ytrain = traitement.l_Ytrain
        self.l_Xtest = traitement.l_Xtest
        self.l_Ytest = traitement.l_Ytest
        self.labels = traitement.labels
        
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
    
    

    # Fonctions d'affichage
    
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