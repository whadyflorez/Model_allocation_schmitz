
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Datos del problema
ni, nj = 3, 3
source_avai = [20, 55, 60]
sink_dem = [10, 11, 12]

# Costos c_ij
c = np.array([
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [1.2, 1.2, 1.2]
])

# Diccionarios de mapeo
dmap = {}
k = 0
for i in range(ni):
    for j in range(nj):
        dmap[k] = (i, j)
        k += 1
idmap = {v: k for k, v in dmap.items()}

# Funciones objetivo
def f1(x):
    return -np.sum(x)  # Queremos maximizar flujo total

def f2(x):
    return sum(c[i,j] * x[idmap[(i,j)]] for i in range(ni) for j in range(nj))

# Restricciones del sistema
def consfun_eq(x):  # igualdades por demanda
    return np.array([
        sum(x[idmap[(i,0)]] for i in range(ni)) - sink_dem[0],
        sum(x[idmap[(i,1)]] for i in range(ni)) - sink_dem[1],
        sum(x[idmap[(i,2)]] for i in range(ni)) - sink_dem[2],
    ])

def consfun_ueq(x):  # ofertas máximas
    return np.array([
        source_avai[0] - sum(x[idmap[(0,j)]] for j in range(nj)),
        source_avai[1] - sum(x[idmap[(1,j)]] for j in range(nj)),
        source_avai[2] - sum(x[idmap[(2,j)]] for j in range(nj)),
    ])

bounds = [(0, None)] * 9
eq_constr = {'type': 'eq', 'fun': consfun_eq}
ueq_constr = {'type': 'ineq', 'fun': consfun_ueq}

# Exploración de restricciones sobre el costo (f2)
eps_vals = np.linspace(0, 130, 100)
pareto = []
x_sols = []


for eps in eps_vals:
    cons_eps = {'type': 'ineq', 'fun': lambda x, e=eps: e - f2(x)}
    res = minimize(f1, x0=np.ones(9), method='SLSQP',
                   bounds=bounds, constraints=[eq_constr, ueq_constr, cons_eps])

    if res.success:
        pareto.append((-f1(res.x), f2(res.x)))  # f1 se vuelve positivo
        x_sols.append(res.x.copy())

pareto = np.array(pareto)
x_sols = np.array(x_sols)

# Verificación y curva de Pareto
if pareto.ndim != 2 or pareto.shape[0] == 0:
    print("❌ No se encontraron soluciones factibles para los valores de epsilon dados.")
    import sys
    sys.exit()

plt.figure(figsize=(7,5))
plt.plot(pareto[:,1], pareto[:,0], 'o-', label="Curva de Pareto")
plt.xlabel("f2: Costo total (restricción)")
plt.ylabel("f1: Flujo total (objetivo)")
plt.title("Frontera de Pareto (restricción en el costo)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Guardar soluciones en tensor
x_tensores = np.zeros((len(x_sols), ni, nj))
for s in range(len(x_sols)):
    for k, (i,j) in dmap.items():
        x_tensores[s, i, j] = x_sols[s][k]

# Mostrar ejemplo de solución
print("\n📌 Ejemplo de solución Pareto (último punto):")
print(f"  Flujo total (f1) = {pareto[-1,0]:.2f}, Costo total (f2) = {pareto[-1,1]:.2f}")
print("  Matriz de flujos x[i,j]:")
print(np.round(x_tensores[-1], 2))
