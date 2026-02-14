import lab2Functions as f
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Datasets that failed in Task 1 without slack
datasets = ['hard']
# More values of slack parameter C
C_values = [1.0, 10.0, 100.0, 1000.0] 

plt.figure(figsize=(17, 4))

plot_idx = 1
for level in datasets:
    for C in C_values:
        N, inputs, targets = f.choose_data(level)
        
        # Precompute the P matrix for efficiency 
        P = np.array([[targets[i]*targets[j]*f.linearKernel(inputs[i], inputs[j]) 
                       for j in range(N)] for i in range(N)])
        
        # Objective function implementing Eq (4) 
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

        # Equality constraint: sum(alpha_i * t_i) = 0
        def zerofun(alpha):
            return np.dot(alpha, targets)
        
        # Soft margin constraint: 0 <= alpha <= C 
        B = [(0, C) for _ in range(N)]
        ret = minimize(objective, np.zeros(N), bounds=B, 
                       constraints={'type':'eq', 'fun':zerofun})

        ax = plt.subplot(1, 4, plot_idx)
        classA, classB = inputs[targets == 1], inputs[targets == -1]
        ax.plot(classA[:, 0], classA[:, 1], 'b.', label='Class A')
        ax.plot(classB[:, 0], classB[:, 1], 'r.', label='Class B')

        if ret['success']:
            alphas = ret['x']
            # Support Vectors: alpha > threshold 
            sv_idx = np.where(alphas > 1e-5)[0]
            s_alphas, s_inputs, s_targets = alphas[sv_idx], inputs[sv_idx], targets[sv_idx]
            
            # Critical: b must be calculated using a point on the margin 
            # A point is on the margin if 0 < alpha < C 
            margin_svs = np.where((alphas > 1e-5) & (alphas < C * 0.999))[0]
            
            if len(margin_svs) > 0:
                ref = margin_svs[0]
                b = f.indicator(inputs[ref], s_alphas, s_targets, s_inputs, 0, f.linearKernel) - targets[ref]
            else:
                # Fallback to the first SV if none are strictly on the margin
                b = f.indicator(s_inputs[0], s_alphas, s_targets, s_inputs, 0, f.linearKernel) - s_targets[0]

            # Mark all SVs (circles) 
            ax.scatter(s_inputs[:, 0], s_inputs[:, 1], s=80, facecolors='none', edgecolors='k', label='SVs')
            
            # Plot margins and boundary 
            xgrid = np.linspace(-4, 4, 100)
            ygrid = np.linspace(-4, 4, 100)
            grid = np.array([[f.indicator(np.array([x, y]), s_alphas, s_targets, s_inputs, b, f.linearKernel) 
                              for x in xgrid] for y in ygrid])
            
            ax.contour(xgrid, ygrid, grid, levels=[-1.0, 0.0, 1.0], 
                        colors=('red', 'black', 'blue'), 
                        linestyles=('dashed', 'solid', 'dashed'), linewidths=(1, 2, 1))
            
            ax.set_title(f"C={C}")
        else:
            ax.set_title(f"C={C} - Failed")

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        plot_idx += 1

plt.suptitle("Task 4: Linear SVM with Different C Values", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()