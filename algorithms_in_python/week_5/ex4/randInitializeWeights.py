import numpy as np

def randInitializeWeights(L_in, L_out):
    """Randomly initialize the weights of a layer with L_in
       incoming connections and L_out outgoing connections
       W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
       of a layer with L_in incoming connections and L_out outgoing
       connections.
       Note that W should be set to a matrix of size(L_out, 1 + L_in) as
       the first row of W handles the "bias" terms
    """
        
    epsilon_init = 0.12
    W = np.random.uniform(0, 1, (L_out, L_in + 1)) * 2 * epsilon_init - epsilon_init
    
    return W

