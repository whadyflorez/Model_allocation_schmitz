import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Objetivos (minimizar)
c1 = np.array([3.0, 1.0])   # f1(x) = 3 x1 +  x2
c2 = np.array([1.0, 2.0])   # f2(x) = 1 x1 + 2 x2

# Restricciones Ax <= b; x >= 0
# Límites de recursos + cotas superiores + "demanda" x1 + x2 >= 5
A = np.array([
    [2.0, 1.0],   # 2x1 + x2  <= 14
    [1.0, 2.0],   #  x1 + 2x2 <= 14
    [1.0, 0.0],   #  x1       <= 6
    [0.0, 1.0],   #       x2  <= 6
    [-1.0, -1.0], # -(x1+x2) <= -5  -> x1 + x2 >= 5
])
b = np.array([14.0, 14.0, 6.0, 6.0, -5.0])

def solve_lp_minimize(c, A, b, method='highs-ds'):
    res = linprog(c=c, A_ub=A, b_ub=b, bounds=[(0, None)]*A.shape[1], method=method)
    return None if not res.success else {"x": res.x, "f": c @ res.x}

def augment_with_epsilon_constraint(A, b, c2, eps):
    A_aug = np.vstack([A, c2])
    b_aug = np.hstack([b, eps])
    return A_aug, b_aug

# Extremos para el rango de eps: min f2 y f2 en el óptimo de f1
ext_f2 = solve_lp_minimize(c2, A, b)              # da x ≈ [5, 0], f2_min = 5
ext_f1 = solve_lp_minimize(c1, A, b)              # da x ≈ [0, 5], f2 en ese punto = 10

eps_start = ext_f2["f"]
eps_end   = max(eps_start, c2 @ ext_f1["x"])

# Barrido ε-constraint
eps_values = np.linspace(eps_start, eps_end, 21)
pareto = []
for eps in eps_values:
    A_eps, b_eps = augment_with_epsilon_constraint(A, b, c2, eps)
    sol = solve_lp_minimize(c1, A_eps, b_eps)
    if sol is None: 
        continue
    x = sol["x"]
    pareto.append((c1 @ x, c2 @ x, x))

# Orden y reporte
pareto = sorted({(round(f1,8), round(f2,8), tuple(np.round(x,8))) for f1,f2,x in pareto}, key=lambda t:t[1])

print("Extremo (min f2): x* =", ext_f2["x"], " f1 =", c1 @ ext_f2["x"], " f2 =", ext_f2["f"])
print("Extremo (min f1): x* =", ext_f1["x"], " f1 =", ext_f1["f"], " f2 =", c2 @ ext_f1["x"])
print("\nPuntos Pareto (ε-constraint):")
for f1, f2, x in pareto:
    print(f"  f1={f1:6.2f}  f2={f2:6.2f}   x={x}")

# Plot del frente (f2 en eje x, f1 en eje y)
f1_vals = [p[0] for p in pareto]
f2_vals = [p[1] for p in pareto]
plt.figure()
plt.plot(f2_vals, f1_vals, marker='o')
plt.xlabel(r'$f_2(x)=x_1+2x_2$')
plt.ylabel(r'$f_1(x)=3x_1+x_2$')
plt.title('Frente de Pareto (ε-constraint + HiGHS Dual Simplex)')
plt.grid(True)
plt.show()
