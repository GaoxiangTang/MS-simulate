# optimize for potential that can trap a uniform ion chain

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
from scipy.misc import derivative
import numdifftools as nd
from scipy import constants


def Energy(N, u, order=2, gamma=None):
    if order==4:
        E=(-0.5*u**2+0.25*gamma*u**4).sum()
    if order==2:
        E=(0.5*u**2).sum()
    for i in range(N):
        for j in range(i):
            E+=1/abs(u[i]-u[j])
    return E


def position(N, order=2, gamma=None):
    x0=np.linspace(-1, 1, N)
    return np.array(sorted(minimize(lambda x: Energy(N, x, order, gamma=gamma), x0).x))

def optimize_for_gamma(N, ):
    def RSD(u):
        du = np.diff(u[1:-1])
        return np.std(du)/du.mean()
    
    g = np.linspace(0.5 ,10, 100)
    plt.plot(g, [RSD(position(N, order=4, gamma=gg)) for gg in g])
    plt.show()
    
    return minimize_scalar(lambda g: RSD(position(N, order=4, gamma=g)), bounds=[0.5, 10]).x