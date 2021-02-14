import model_abstract as baseM

class model_mix_gauss(baseM.base_model):
    def __init__(self):

    def fit(self, datax, datay = None):
        '''
        param: datax : [[[vector_vitesse_absci, vector_vitesse_ord]*N]*nb_traj]
        '''
    
    def EM_main(self, iterate = 100):
    
    def EM_init(self):
        

    
    def predict(self, x_test):
        pass
    
    def score(self, func_test, x_test):
        return func_test(x_test)     