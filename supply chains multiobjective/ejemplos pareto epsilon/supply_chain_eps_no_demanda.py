
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ni, nj = 3, 3
source_avai = [20, 55, 60]

c = np.array([
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [1.2, 1.2, 1.2]
])

dmap = {k: (i, j) for k, (i, j) in enumerate([(i, j) for i in range(ni) for j in range(nj)])}
idmap = {v: k for k, v in dmap.items()}

def f1(x):
    return -np.sum(x)  # Maximizar flujo total

def f2(x):
    return sum(c[i, j] * x[idmap[(i, j)]] for i in range(ni) for j in range(nj))

def consfun_ueq(x):
    return np.array([
        source_avai[0] - sum(x[idmap[(0,j)]] for j in range(nj)),
        source_avai[1] - sum(x[idmap[(1,j)]] for j in range(nj)),
        source_avai[2] - sum(x[idmap[(2,j)]] for j in range(nj)),
    ])

bounds = [(0, None)] * 9
ueq_constr = {'type': 'ineq', 'fun': consfun_ueq}

eps_vals = np.linspace(60, 140, 25)
pareto = []
x_sols = []

for eps in eps_vals:
    cons_eps = {'type': 'ineq', 'fun': lambda x, e=eps: e - f2(x)}
    res = minimize(f1, x0=np.ones(9), method='SLSQP',
                   bounds=bounds, constraints=[ueq_constr, cons_eps],
                   options={'ftol': 1e-6, 'disp': False})

    if res.success:
        pareto.append((-f1(res.x), f2(res.x)))
        x_sols.append(res.x.copy())

pareto = np.array(pareto)

if pareto.ndim != 2 or pareto.shape[0] == 0:
    print("❌ No se encontraron soluciones factibles.")
    import sys
    sys.exit()

plt.figure(figsize=(7, 5))
plt.plot(pareto[:,1], pareto[:,0], 'o-', label="Sin restricción de demanda")
plt.xlabel("f2: Costo total (restricción)")
plt.ylabel("f1: Flujo total (objetivo)")
plt.title("Curva de Pareto real (sin restricciones de demanda)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
