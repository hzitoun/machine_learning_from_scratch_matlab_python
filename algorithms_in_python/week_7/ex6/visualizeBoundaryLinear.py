import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData


def visualizeBoundaryLinear(X, y, model):
    """Plots a linear decision boundary learned by the SVM
        Plots a linear decision boundary 
        learned by the SVM and overlays the data on it
    """
    
    w = model.w
    b = model.b
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = - (w[0]*xp + b)/w[1]
    plotData(X, y)
    plt.plot(xp, yp, color='blue')
