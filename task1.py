import lab2Functions as f
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

datasets = ['easy', 'hard', 'very_hard']
plt.figure(figsize=(18, 7))


for idx, level in enumerate(datasets):
    N, inputs, targets = f.choose_data(level)
    P = np.array([[targets[i]*targets[j]*f.linearKernel(inputs[i], inputs[j]) for j in range(N)] for i in range(N)])
    
    def objective(alpha):
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

    def zerofun(alpha):
        return np.dot(alpha, targets)
    
    B= bounds=[(0, None) for _ in range(N)]
    ret = minimize(objective, np.zeros(N), bounds=B, constraints={'type':'eq', 'fun':zerofun})

    plt.subplot(1, 3, idx+1)
    classA, classB = inputs[targets == 1], inputs[targets == -1]
    plt.plot(classA[:, 0], classA[:, 1], 'b.', label='Class A')
    plt.plot(classB[:, 0], classB[:, 1], 'r.', label='Class B')

    if ret['success']:
        alphas = ret['x']
        sv_idx = np.where(alphas > 1e-5)[0]
        s_alphas, s_inputs, s_targets = alphas[sv_idx], inputs[sv_idx], targets[sv_idx]
        b = f.indicator(s_inputs[0], s_alphas, s_targets, s_inputs, 0, f.linearKernel) - s_targets[0]
        print(f"Dataset: {level}, Support Vectors: {len(s_alphas)}, Bias: {b:.4f}")
        
        plt.scatter(s_inputs[:, 0], s_inputs[:, 1], s=80, facecolors='none', edgecolors='k', label='SVs')
        xgrid, ygrid = np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)
        grid = np.array([[f.indicator(np.array([x, y]), s_alphas, s_targets, s_inputs, b, f.linearKernel) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid, levels=[-1.0, 0.0, 1.0], colors=('red', 'black', 'blue'), linestyles=('dashed', 'solid', 'dashed'), linewidths=(1, 2, 1))
    else:
        plt.text(0, 0, "No linear separation possible", ha='center', color='red', weight='bold')
        print(f"Dataset: {level}, Optimization Failed") 
    
    plt.title(f"Linear - {level}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend()

plt.suptitle("SVM with Linear Kernel on Different Datasets withou slack", fontsize=16)
plt.tight_layout()
plt.show()