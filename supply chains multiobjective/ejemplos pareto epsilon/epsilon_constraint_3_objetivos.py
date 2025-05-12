
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Objetivos
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

def f3(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

# Restricciones comunes
def constraint1(x):
    return 3 - (x[0] + x[1])  # x1 + x2 <= 3

def constraint2(x):
    return x[0]               # x1 >= 0

def constraint3(x):
    return x[1]               # x2 >= 0

# Rejilla de valores epsilon para f2 y f3
epsilons_f2 = np.linspace(0.5, 6.0, 10)
epsilons_f3 = np.linspace(0.5, 6.0, 10)

pareto_points = []
solutions = []

# Recorremos la rejilla 2D de eps_f2 y eps_f3
for eps2 in epsilons_f2:
    for eps3 in epsilons_f3:
        constraints = [
            {"type": "ineq", "fun": constraint1},
            {"type": "ineq", "fun": constraint2},
            {"type": "ineq", "fun": constraint3},
            {"type": "ineq", "fun": lambda x, e=eps2: e - f2(x)},
            {"type": "ineq", "fun": lambda x, e=eps3: e - f3(x)}
        ]
        res = minimize(f1, x0=[0.5, 0.5], constraints=constraints)
        if res.success:
            f1_val = f1(res.x)
            f2_val = f2(res.x)
            f3_val = f3(res.x)
            pareto_points.append([f1_val, f2_val, f3_val])
            solutions.append([*res.x])

pareto_points = np.array(pareto_points)
solutions = np.array(solutions)

# Gráfica en 3D de la frontera de Pareto aproximada
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
           c='blue', marker='o', label='Frontera de Pareto (ε-constraint)')
ax.set_xlabel("f1(x)")
ax.set_ylabel("f2(x)")
ax.set_zlabel("f3(x)")
ax.set_title("Aproximación de la frontera de Pareto (3 objetivos)")
ax.legend()
plt.tight_layout()
plt.show()

# Mostrar soluciones
print("\n🧾 Soluciones óptimas:")
for i in range(len(solutions)):
    x = solutions[i]
    f = pareto_points[i]
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}) → f1 = {f[0]:.4f}, f2 = {f[1]:.4f}, f3 = {f[2]:.4f}")
