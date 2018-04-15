import numpy as np

def debugInitializeWeights(fan_out, fan_in):
    """ Initialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging
    W = debugInitializeWeights(fan_in, fan_out) initializes the weights 
    of a layer with fan_in incoming connections and fan_out outgoing 
    connections using a fix set of values"""
    
    # Set W to zeros
    W = np.zeros((fan_out, fan_in + 1))
    
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging   
    array = np.sin(np.arange(W.size))
    
    W = np.reshape(array, W.shape) / 10
    
    return W


