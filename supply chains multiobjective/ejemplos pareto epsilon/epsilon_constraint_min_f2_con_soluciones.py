
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Objetivos
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

# Restricciones comunes
def constraint1(x):
    return 3 - (x[0] + x[1])  # x1 + x2 <= 3

def constraint2(x):
    return x[0]               # x1 >= 0

def constraint3(x):
    return x[1]               # x2 >= 0

# Rango de epsilon para f1 ahora
epsilons = np.linspace(0.5, 6.0, 25)
pareto_points = []
solutions = []

# Optimización para cada epsilon
for eps in epsilons:
    constraints = [
        {"type": "ineq", "fun": constraint1},
        {"type": "ineq", "fun": constraint2},
        {"type": "ineq", "fun": constraint3},
        {"type": "ineq", "fun": lambda x, e=eps: e - f1(x)}  # f1(x) <= eps
    ]
    res = minimize(f2, x0=[0.5, 0.5], constraints=constraints)
    if res.success:
        pareto_points.append([f1(res.x), f2(res.x)])
        solutions.append(res.x)

pareto_points = np.array(pareto_points)
solutions = np.array(solutions)

# Gráfica de la frontera de Pareto
plt.figure(figsize=(8, 6))
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'o-', label="Frontera de Pareto (min f2, f1 ≤ ε)")
plt.xlabel("f1(x)")
plt.ylabel("f2(x)")
plt.title("Frontera de Pareto (ε-restricción sobre f1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Imprimir soluciones
print("\n🧾 Soluciones óptimas (x1, x2):")
for i, x in enumerate(solutions):
    print(f"ε = {epsilons[i]:.2f} → x = ({x[0]:.4f}, {x[1]:.4f})")
