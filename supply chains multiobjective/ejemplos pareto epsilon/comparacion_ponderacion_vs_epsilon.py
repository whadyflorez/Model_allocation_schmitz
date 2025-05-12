
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Objetivos
def f1(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def f2(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

# Restricciones comunes
def constraint1(x):
    return 3 - (x[0] + x[1])
def constraint2(x):
    return x[0]
def constraint3(x):
    return x[1]

# ---------- MÉTODO 1: PONDERACIÓN ----------
weights = np.linspace(0, 1, 25)
pareto_weighted = []

for w in weights:
    def f_weighted(x):
        return w * f1(x) + (1 - w) * f2(x)

    constraints = [
        {"type": "ineq", "fun": constraint1},
        {"type": "ineq", "fun": constraint2},
        {"type": "ineq", "fun": constraint3},
    ]
    res = minimize(f_weighted, x0=[0.5, 0.5], constraints=constraints)
    if res.success:
        pareto_weighted.append([f1(res.x), f2(res.x)])

pareto_weighted = np.array(pareto_weighted)

# ---------- MÉTODO 2: ε-RESTRICCIÓN ----------
epsilons = np.linspace(0.5, 6.0, 25)
pareto_epsilon = []

for eps in epsilons:
    constraints = [
        {"type": "ineq", "fun": constraint1},
        {"type": "ineq", "fun": constraint2},
        {"type": "ineq", "fun": constraint3},
        {"type": "ineq", "fun": lambda x, e=eps: e - f2(x)}
    ]
    res = minimize(f1, x0=[0.5, 0.5], constraints=constraints)
    if res.success:
        pareto_epsilon.append([f1(res.x), f2(res.x)])

pareto_epsilon = np.array(pareto_epsilon)

# ---------- GRÁFICA COMPARATIVA ----------
plt.figure(figsize=(8, 6))
plt.plot(pareto_weighted[:, 0], pareto_weighted[:, 1], 'o-', label="Ponderación")
plt.plot(pareto_epsilon[:, 0], pareto_epsilon[:, 1], 's--', label="ε-Restricción")
plt.xlabel("f1(x)")
plt.ylabel("f2(x)")
plt.title("Comparación: Ponderación vs. ε-Restricción")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
