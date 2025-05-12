
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

ni = 3  # sources
nj = 3  # sinks

source_avai = [20, 55, 60]
sink_dem = [10, 11, 12]

# Diccionario de mapeo
k = 0
dmap = {}
for i in range(ni):
    for j in range(nj):
        dmap[k] = (i, j)
        k += 1
idmap = {value: key for key, value in dmap.items()}

# Restricciones de igualdad: satisfacer demandas
def consfun_eq(x):
    y = np.zeros(3)
    y[0] = x[idmap[(0,0)]] + x[idmap[(1,0)]] + x[idmap[(2,0)]]
    y[1] = x[idmap[(0,1)]] + x[idmap[(1,1)]] + x[idmap[(2,1)]]
    y[2] = x[idmap[(0,2)]] + x[idmap[(1,2)]] + x[idmap[(2,2)]]
    return y

# Restricciones de desigualdad: no exceder oferta
def consfun_ueq(x):
    y = np.zeros(3)
    y[0] = x[idmap[(0,0)]] + x[idmap[(0,1)]] + x[idmap[(0,2)]]
    y[1] = x[idmap[(1,0)]] + x[idmap[(1,1)]] + x[idmap[(1,2)]]
    y[2] = x[idmap[(2,0)]] + x[idmap[(2,1)]] + x[idmap[(2,2)]]
    return y

# Función objetivo
def fun(x):
    return (
        x[idmap[(0,0)]] + x[idmap[(0,1)]] + x[idmap[(0,2)]] +
        2.0 * (x[idmap[(1,0)]] + x[idmap[(1,1)]] + x[idmap[(1,2)]]) +
        1.2 * (x[idmap[(2,0)]] + x[idmap[(2,1)]] + x[idmap[(2,2)]])
    )

# Límites de variables
bounds = [(0, None)] * 9

# Restricciones
constraints_eq = NonlinearConstraint(consfun_eq, sink_dem, sink_dem)
constraints_ueq = NonlinearConstraint(consfun_ueq, [0]*3, source_avai)
constr_list = [constraints_eq, constraints_ueq]

# Resolución del problema
x0 = np.ones(9)
result = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=constr_list)

# Procesamiento del resultado
x = result.x
x_matrix = np.zeros((ni, nj))
for i in range(ni):
    for j in range(nj):
        x_matrix[i, j] = x[idmap[(i, j)]]

# Impresión organizada
print(result.message)
print("✅ Optimización exitosa" if result.success else "❌ Falló la optimización")
print(f"🎯 Costo mínimo total: {result.fun:.4f}\n")



print("\n📊 Suma por sink (demanda atendida):")
for j in range(nj):
    print(f"  Sink {j}: {np.sum(x_matrix[:, j]):.2f}")

print("\n📊 Suma por source (oferta utilizada):")
for i in range(ni):
    print(f"  Source {i}: {np.sum(x_matrix[i, :]):.2f}")
