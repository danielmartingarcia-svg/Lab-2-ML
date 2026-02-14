import lab2Functions as f
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

datasets = ['easy', 'hard', 'very_hard']
rho_values = [2, 3, 5]  
sigma_values = [0.5, 1.0, 2.0]

for level in datasets:
    N, inputs, targets = f.choose_data(level) 
    
    # Polynomial kernel figure
    plt.figure(figsize=(18, 7))
    for rho_idx, rho in enumerate(rho_values):
        kernels = [
            (f.polynomialKernel, (rho,), f"Polynomial (p={rho})")
        ]

        for k_func, k_args, name in kernels:
            # Precompute P inside the loop for the specific kernel
            P = np.array([[targets[i]*targets[j]*k_func(inputs[i], inputs[j], *k_args) for j in range(N)] for i in range(N)])
            
            def objective(alpha):
                return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

            def zerofun(alpha):
                return np.dot(alpha, targets)
            
            ret = minimize(objective, np.zeros(N), bounds=[(0, None) for _ in range(N)], 
                        constraints={'type':'eq', 'fun':zerofun})

            plt.subplot(1, 3, rho_idx+1)
            classA, classB = inputs[targets == 1], inputs[targets == -1]
            plt.plot(classA[:, 0], classA[:, 1], 'b.', label='Class A')
            plt.plot(classB[:, 0], classB[:, 1], 'r.', label='Class B')

            if ret['success']:
                alphas = ret['x']
                sv_idx = np.where(alphas > 1e-5)[0]
                s_alphas, s_inputs, s_targets = alphas[sv_idx], inputs[sv_idx], targets[sv_idx]
                # Calculate b using a support vector
                b = f.indicator(s_inputs[0], s_alphas, s_targets, s_inputs, 0, k_func, *k_args) - s_targets[0]
                print(f"Kernel: {name}, dataset {level}, Support Vectors: {len(s_alphas)}, Bias: {b:.4f}")
                
                plt.scatter(s_inputs[:, 0], s_inputs[:, 1], s=80, facecolors='none', edgecolors='k', label='SVs')
                xgrid, ygrid = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
                grid = np.array([[f.indicator(np.array([x, y]), s_alphas, s_targets, s_inputs, b, k_func, *k_args) for x in xgrid] for y in ygrid])
                plt.contour(xgrid, ygrid, grid, levels=[-1.0, 0.0, 1.0], colors=('red', 'black', 'blue'), 
                            linestyles=('dashed', 'solid', 'dashed'), linewidths=(1, 2, 1)) 
            else:
                plt.text(0, 0, "No separation possible", ha='center', color='red', weight='bold')   
                print(f"Kernel: {name}, dataset {level}, Optimization Failed")
                    
            plt.title(f"Kernel: {name}")
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()

    plt.suptitle("Polynomial Kernel for the " + level + " dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

    # RBF kernel figure
    plt.figure(figsize=(18, 7))
    for sigma_idx, sigma in enumerate(sigma_values):    
        kernels = [
            (f.rbfKernel, (sigma,), f"RBF (sigma={sigma})")
        ]

        for k_func, k_args, name in kernels:
            # Precompute P inside the loop for the specific kernel
            P = np.array([[targets[i]*targets[j]*k_func(inputs[i], inputs[j], *k_args) for j in range(N)] for i in range(N)])
            
            def objective(alpha):
                return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

            def zerofun(alpha):
                return np.dot(alpha, targets)
            
            ret = minimize(objective, np.zeros(N), bounds=[(0, None) for _ in range(N)], 
                        constraints={'type':'eq', 'fun':zerofun})

            plt.subplot(1, 3, sigma_idx+1)
            classA, classB = inputs[targets == 1], inputs[targets == -1]
            plt.plot(classA[:, 0], classA[:, 1], 'b.', label='Class A')
            plt.plot(classB[:, 0], classB[:, 1], 'r.', label='Class B')

            if ret['success']:
                alphas = ret['x']
                sv_idx = np.where(alphas > 1e-5)[0]
                s_alphas, s_inputs, s_targets = alphas[sv_idx], inputs[sv_idx], targets[sv_idx]
                # Calculate b using a support vector
                b = f.indicator(s_inputs[0], s_alphas, s_targets, s_inputs, 0, k_func, *k_args) - s_targets[0]
                print(f"Kernel: {name}, dataset {level}, Support Vectors: {len(s_alphas)}, Bias: {b:.4f}")
                
                plt.scatter(s_inputs[:, 0], s_inputs[:, 1], s=80, facecolors='none', edgecolors='k', label='SVs')
                xgrid, ygrid = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
                grid = np.array([[f.indicator(np.array([x, y]), s_alphas, s_targets, s_inputs, b, k_func, *k_args) for x in xgrid] for y in ygrid])
                plt.contour(xgrid, ygrid, grid, levels=[-1.0, 0.0, 1.0], colors=('red', 'black', 'blue'), 
                            linestyles=('dashed', 'solid', 'dashed'), linewidths=(1, 2, 1))
            else:
                plt.text(0, 0, "No separation possible", ha='center', color='red', weight='bold')   
                print(f"Kernel: {name}, dataset {level}, Optimization Failed")

            plt.title(f"Kernel: {name}")
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()

    plt.suptitle("RBF Kernel for the " + level + " dataset", fontsize=16)
    plt.tight_layout()
    plt.show()        