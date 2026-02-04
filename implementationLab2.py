import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objective(alpha):
    return none # To be Implemented

def minimize(objective, start, bounds, constrains):
    '''
    Returns a dictionary data structure
    'success': True/False¨
    'x': optimal values for alpha
    '''
    return none # To be Implemented

def linearKernel(x, y):
    return np.dot(x, y)   

def polynomialKernel(x, y, p):
    return (1 + np.dot(x, y)) ** p

def rbfKernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def zerofun():
    '''
    Impose the equality constraint sum(alpha[i] * t[i]) = 0
    Takes a vector as input and returns a scalar value which should be constrained to zero
    '''
    # scalar = [expr for x in seq]  something like this
    return  none # To be Implemented