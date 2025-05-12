import numpy as np
from scipy.optimize import minimize

I, J, K = 3, 3, 3

D = np.array([
    [20, 30, 10],
    [15, 20, 25],
    [30, 15, 20],
])

P = np.array([
    [0.4, 0.4],
    [0.5, 0.3],
    [0.6, 0.2]
])

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

for i in range(I):
    for k_ in range(K):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i, k_=k_: D[i, k_] - sum(x[idmap[(i, j, k_)]] for j in range(J))
        })

for j in range(J):
    for m in range(K - 1):
        constraints.append({
            'type': 'eq',
            'fun': lambda x, j=j, m=m: (
                sum(x[idmap[(i, j, m)]] for i in range(I)) -
                P[j, m] * sum(x[idmap[(i, j, k)]] for i in range(I) for k in range(K))
            )
        })

# Flujo mínimo por destino
T_min = [10.0, 10.0, 10.0]
for j in range(J):
    constraints.append({
        'type': 'ineq',
        'fun': lambda x, j=j: sum(x[idmap[(i, j, k)]] for i in range(I) for k in range(K)) - T_min[j]
    })

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
    x_tensor = result.x.reshape((I, J, K))  # Guardar resultados como tensor
    print(f"\n✅ Flujo total máximo: {total:.4f}")
    for k, (i, j, l) in dmap.items():
        val = result.x[k]
        if val > 1e-6:
            print(f"Source {i} → Sink {j}, producto {l}: {val:.2f}")
else:
    print("\n❌ Optimización fallida:", result.message)
    
    



print("\n🔎 Análisis de proporciones por destino:")
for j in range(J):
    total_sink = x_tensor[:, j, :].sum()
    if total_sink > 0:
        proporciones = x_tensor[:, j, :].sum(axis=0) / total_sink
        esperado = np.append(P[j], 1 - np.sum(P[j]))
        error = np.abs(proporciones - esperado)
        print(f"Sink {j}: Total = {total_sink:.2f}")
        print(f"  Proporciones reales: {proporciones.round(4)}")
        print(f"  Esperadas           : {esperado.round(4)}")
        print(f"  Error absoluto      : {error.round(4)}\n")
    else:
        print(f"Sink {j}: No recibió ningún flujo.\n")
