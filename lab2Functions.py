import numpy as np
import random

def linearKernel(x1, x2):
    return np.dot(x1, x2) 

def polynomialKernel(x1, x2, p=3):
    return (np.dot(x1, x2) + 1) ** p

def rbfKernel(x1, x2, sigma=1.0):
    diff = x1 - x2
    return np.exp(-np.dot(diff, diff) / (2 * sigma**2)) 

# Helper for the indicator function
def indicator(s, s_alphas, s_targets, s_inputs, b, kernel_func, *args):
    # Map the kernel function over all support vectors for the new point s
    # This replaces the explicit 'for' loop with a list comprehension/array
    ks = np.array([kernel_func(s, x_i, *args) for x_i in s_inputs])
    
    # Use np.dot for a high-speed scalar product of the weights and kernel results
    return np.dot(s_alphas * s_targets, ks) - b

def choose_data(easyness):
    # Force global seeds for reproducibility
    np.random.seed(100)
    random.seed(100)
    
    scale = {'easy': 0.2, 'hard': 0.3, 'very_hard': 0.5}[easyness]
    
    classA = np.concatenate((
        np.random.randn(10, 2) * scale + [1.5, 0.5],
        np.random.randn(10, 2) * scale + [-1.5, 0.5]
    ))
    classB = np.random.randn(20, 2) * scale + [0.0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(len(classA)), -np.ones(len(classB))))
    
    N = inputs.shape[0]
    permute = list(range(N))
    random.shuffle(permute)
    return N, inputs[permute], targets[permute]