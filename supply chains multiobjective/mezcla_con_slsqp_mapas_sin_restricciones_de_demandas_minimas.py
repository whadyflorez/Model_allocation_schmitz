import numpy as np
from scipy.optimize import minimize

I, J, K = 3, 3, 3  # sources, sinks, products

# Disponibilidad D_ik
D = np.array([
    [20, 30, 10],
    [15, 20, 25],
    [30, 15, 20],
])

# Composición deseada (solo K-1 columnas necesarias)
P = np.array([
    [0.4, 0.4],
    [0.5, 0.3],
    [0.6, 0.2]
])

# Diccionario para mapear variables x_{ijk} a un índice lineal
dmap = {}
k = 0
for i in range(I):
    for j in range(J):
        for l in range(K):
            dmap[k] = (i, j, l)
            k += 1

idmap = {v: k for k, v in dmap.items()}
n_vars = len(dmap)

def objective(x):
    return -np.sum(x)

constraints = []

# (1) Disponibilidad: ∑_j x[i,j,k] ≤ D[i,k]
for i in range(I):
    for k_ in range(K):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i, k_=k_: D[i, k_] - sum(x[idmap[(i, j, k_)]] for j in range(J))
        })

# (2) Proporciones fijas en cada destino j, solo para K-1 productos
for j in range(J):
    for m in range(K - 1):
        constraints.append({
            'type': 'eq',
            'fun': lambda x, j=j, m=m: (
                sum(x[idmap[(i, j, m)]] for i in range(I)) -
                P[j, m] * sum(x[idmap[(i, j, k)]] for i in range(I) for k in range(K))
            )
        })

# Cotas
bounds = [(0, None)] * n_vars
x0 = np.ones(n_vars)

result = minimize(
    fun=objective,
    x0=x0,
    bounds=bounds,
    constraints=constraints,
    method='SLSQP',
    options={'disp': True, 'ftol': 1e-9, 'maxiter': 1000}
)

if result.success:
    total = -result.fun
    print(f"\n✅ Flujo total máximo: {total:.4f}")
    for k, (i, j, l) in dmap.items():
        val = result.x[k]
        if val > 1e-6:
            print(f"Source {i} → Sink {j}, producto {l}: {val:.2f}")
else:
    print("\n❌ Optimización fallida:", result.message)
