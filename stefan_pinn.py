import numpy as np
import matplotlib.pyplot as plt 
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin, sqrt, exp
import random
from tensorflow.keras.callbacks import EarlyStopping
import os
import time
import math

def stefanPINN(nLayer, nNeuron, nEpoch = 10000, earlyStopping = None, alpha = 1.0, t0 = 0.1, TOLX=0.004, TOLT=0.002):
    
    # Inital
    s0 = alpha * t0

    # Variable definition
    x = sn.Variable('x')
    t = sn.Variable('t')
    u = sn.Functional (["u"], [x, t], nLayer*[nNeuron] , 'tanh')
    s = sn.Functional (["s"], [t], nLayer*[nNeuron] , 'tanh')

    # Diff. equation, heat
    L1 =  diff(u, t) - alpha * diff(u, x, order=2)

    # Stefan condition
    C1 = (1/alpha*diff(s, t) + diff(u,x)) * (1 + sign(x - (s-TOLX))) * (1-sign(x-s))

    # Initial s for t=t0
    C2 = (1-sign(t - (t0+TOLT))) * ( s - s0 )

    # Boundary condition "u" when x=0
    C3 = (1-sign(x - (0 +TOLX))) *  ( u - exp(alpha*t) )

    # The temperature at the boundary between the phases is 1
    C4 = (1-sign(x - (s+TOLX))) * (1+sign(x-s)) * (u-1)

    x_data, t_data = [], []

    # Training set
    x_train, t_train = np.meshgrid(
        np.linspace(0, 1, 300),
        np.linspace(t0, 0.5, 300)
    )

    x_data, t_data = np.array(x_train), np.array(t_train)

    m = sn.SciModel([x, t], [L1,C1,C2,C3,C4], 'mse', 'Adam')

    callbacks = [] 
    if earlyStopping != None:
        callbacks = [earlyStopping]

    history = m.train(  [x_data, t_data], 
                        5*['zero'], 
                        learning_rate=0.002, 
                        batch_size=1024, 
                        epochs=nEpoch, 
                        callbacks=callbacks)

    # Test
    x_test, t_test = np.meshgrid(
        np.linspace(0, 1, 300), 
        np.linspace(0.01, 1, 300)
    )

    u_pred = u.eval(m, [x_test, t_test])
    s_pred = s.eval(m, [x_test, t_test])

    s=[]
    for e in s_pred:
        s.append(e[0])

    u_target=[]
    for e in x_test[0]:
        u_target.append(math.exp(alpha*0.5-e))
    
    createFigure(t_test[:,0], s, "PINN", t_test[:,0] * alpha, "Target", "","t", "s", "s")
    createFigure(x_test[149], u_pred[149], "u(0.5) - PINN", u_target, "u(0.5) - target", "","x/s", "Temp. [c]", "temp")

def createFigure(x, prediction, predictionLabel, target, targetLabel, title, xLabel, ylabel, figureName):
    fig = plt.figure()
    plt.plot(x, prediction, label=predictionLabel)
    plt.plot(x, target, label=targetLabel)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(figureName + '.png') 
    plt.close()

if __name__ == "__main__":
    earlyStopping = EarlyStopping(monitor='loss', patience = 200, restore_best_weights = True)
    stefanPINN(nLayer = 4, nNeuron = 20, nEpoch = 10000, earlyStopping = earlyStopping, alpha = 1.0, t0 = 0.1, TOLX=0.004, TOLT=0.002)
