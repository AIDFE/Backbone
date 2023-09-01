from sacred import Experiment           # Sacred 相关
ex = Experiment('MedAI')         # Sacred 相关

@ex.config          # Sacred 相关
def cfg():
    C = 1.0
    gamma = 0.7


@ex.automain        # Sacred 相关
def run(C, gamma):
    
    return C, gamma
