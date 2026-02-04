import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# Set the slack parameter C here.
# Low C = High Bias, Low Variance (more slack, wider margin).
# High C = Low Bias, High Variance (hard margin, strict).
C_VALUE = 100.0 
KERNEL_TYPE = 'rbf' # 'linear', 'polynomial', or 'rbf'
P_VAL = 3              # Degree for polynomial kernel
SIGMA = 0.5            # Sigma for RBF kernel

# --- 2. DATA GENERATION [cite: 103] ---
np.random.seed(100) # Reproducibility [cite: 110]

classA = np.concatenate((
    np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]
))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((
    np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])
))

N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# --- 3. KERNEL FUNCTIONS [cite: 41, 42, 46] ---
def linearKernel(x1, x2):
    return np.dot(x1, x2)

def polynomialKernel(x1, x2, p=P_VAL):
    return (np.dot(x1, x2) + 1) ** p

def rbfKernel(x1, x2, sigma=SIGMA):
    # Euclidean distance squared
    diff = x1 - x2
    dist_sq = np.dot(diff, diff)
    return math.exp(-dist_sq / (2 * sigma**2))

# Select the kernel
def kernel(x1, x2):
    if KERNEL_TYPE == 'linear': return linearKernel(x1, x2)
    elif KERNEL_TYPE == 'polynomial': return polynomialKernel(x1, x2)
    elif KERNEL_TYPE == 'rbf': return rbfKernel(x1, x2)
    else: raise ValueError("Unknown Kernel")

# --- 4. PRE-COMPUTATION (The "P" Matrix)  ---
# We compute t_i * t_j * K(x_i, x_j) once.
print(f"Pre-computing {N}x{N} kernel matrix...")
P = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        P[i, j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])

# --- 5. OPTIMIZATION FUNCTIONS ---

def objective(alpha):
    # Equation 4: 1/2 * alpha^T * P * alpha - sum(alpha)
    # np.dot is matrix multiplication
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

def zerofun(alpha):
    # Equation 5: sum(alpha_i * t_i) = 0
    return np.dot(alpha, targets)

# --- 6. SOLVING ---
# Initial guess: zeros
start = np.zeros(N)

# Bounds: 0 <= alpha <= C [cite: 57]
# If C is None, it's a hard margin (no slack), but usually, we set a large C.
bounds = [(0, C_VALUE) for b in range(N)]

# Constraints [cite: 68]
XC = {'type': 'eq', 'fun': zerofun}

print("Minimizing...")
ret = minimize(objective, start, bounds=bounds, constraints=XC)

if not ret['success']:
    print("WARNING: Optimizer failed to find a solution.")

alpha = ret['x']

# --- 7. EXTRACTING SUPPORT VECTORS [cite: 89] ---
# Threshold to clean up floating point errors
threshold = 1e-5
sv_indices = []

for i in range(N):
    if alpha[i] > threshold:
        sv_indices.append(i)

print(f"Found {len(sv_indices)} support vectors.")

# Store Support Vectors (SVs) separately for easy access
s_alphas = alpha[sv_indices]
s_inputs = inputs[sv_indices]
s_targets = targets[sv_indices]

# --- 8. CALCULATING BIAS (b) [cite: 26] ---
# Equation 7: b = sum(alpha_i * t_i * K(s, x_i)) - t_s
# We only need one SV to calculate b, but averaging over all SVs is more robust numerically.
# Note: We must pick an SV where 0 < alpha < C (on the margin).
# If alpha == C, it's inside the margin (slack active) and equality doesn't strict hold for b calculation.
# For this lab, usually picking the first one works, or averaging.

b = 0
valid_sv_count = 0
for i in range(len(sv_indices)):
    # Calculate b for this specific SV
    sv_idx = sv_indices[i]
    
    # Summation term: sum(alpha_j * t_j * K(x_sv, x_j)) for all training points j
    # (Only non-zero alphas contribute, so we can sum over SVs only)
    k_sum = 0
    for j in range(len(sv_indices)):
        k_sum += s_alphas[j] * s_targets[j] * kernel(s_inputs[i], s_inputs[j])
        
    b_i = k_sum - s_targets[i]
    b += b_i
    valid_sv_count += 1

if valid_sv_count > 0:
    b /= valid_sv_count
else:
    print("Warning: No stable SVs found for b calculation.")

print(f"Calculated bias b = {b}")

# --- 9. INDICATOR FUNCTION [cite: 14, 20] ---
def indicator(x, y):
    point = np.array([x, y])
    k_sum = 0
    # Equation 6: sum(alpha_i * t_i * K(s, x_i)) - b
    for i in range(len(sv_indices)):
        k_sum += s_alphas[i] * s_targets[i] * kernel(point, s_inputs[i])
    return k_sum - b

# --- 10. PLOTTING [cite: 112, 130] ---
plt.figure(figsize=(8, 6))

# Plot data points
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label='Class A')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label='Class B')

# Mark Support Vectors
plt.scatter(s_inputs[:, 0], s_inputs[:, 1], s=80, facecolors='none', edgecolors='k', label='SVs')

# Plot Decision Boundary
xgrid = np.linspace(-4, 4, 100)
ygrid = np.linspace(-4, 4, 100)
grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])

# Contour at -1, 0, 1 (Margins and Boundary)
plt.contour(xgrid, ygrid, grid, levels=[-1.0, 0.0, 1.0], 
            colors=('red', 'black', 'blue'), 
            linestyles=('dashed', 'solid', 'dashed'),
            linewidths=(1, 2, 1))
            

plt.title(f"SVM ({KERNEL_TYPE}) with C={C_VALUE}")
plt.axis('equal')
plt.legend()
plt.show()