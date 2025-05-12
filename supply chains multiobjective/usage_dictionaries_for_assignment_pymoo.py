import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA           # mono-objetivo ver 0.6
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.repair import Repair

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

class TransportRepair(Repair):
    """
    Repara cada individuo para que:
        • Cumpla exactamente las demandas (igualdades columna).
        • No exceda las ofertas (desigualdades fila).
        • Respete xl ≤ x ≤ xu.
    Algoritmo: RAS/IPF con proyección de fila limitada.
    """
    def __init__(self, max_iter=1000, tol=1e-12):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def _do(self, problem, X, **kwargs):

        # accesos rápidos
        id_      = problem.id
        demand   = problem.demand
        supply   = problem.supply
        xl, xu   = problem.xl, problem.xu
        m, n     = 3, 3                      # filas, columnas

        # Repara cada individuo por separado
        for k in range(len(X)):
            x = X[k].copy()

            # ----------- RAS / IPF -----------
            for _ in range(self.max_iter):

                # (1) Escala columnas → demandas EXACTAS
                cols = x.reshape(m, n).sum(axis=0)
                cols[cols == 0] = 1          # evita /0
                x = x.reshape(m, n) * (demand / cols)[None, :]
                x = x.ravel()

                # (2) Escala filas → ofertas PERO con límite superior
                rows = x.reshape(m, n).sum(axis=1)
                scale = np.minimum(1.0, supply / np.where(rows == 0, 1, rows))
                x = (x.reshape(m, n) * scale[:, None]).ravel()

                # (3) Convergencia
                col_ok  = np.allclose(x.reshape(m, n).sum(axis=0),
                                      demand, atol=self.tol)
                row_ok  = np.all(x.reshape(m, n).sum(axis=1) <= supply + self.tol)
                if col_ok and row_ok:
                    break

            # Clip a cotas permitidas
            x = np.clip(x, xl, xu)
            X[k] = x

        return X
# ---------- 1) Definición del problema ----------
class TransportProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=9,            # 3×3 flujos
                         n_obj=1,
                         n_constr=6,         # 3 eq. (~) + 3 ineq.
                         xl=np.zeros(9),
                         xu=np.full(9, 1e3)) # cota superior holgada

        # Índices útiles
        self.id = {(i, j): 3*i + j for i in range(3) for j in range(3)}
        self.demand = np.array([10, 11, 12])
        self.supply = np.array([20, 55, 60])

    # -------- 2) Evaluación ----------
    def _evaluate(self, x, out, *args, **kwargs):

        # Coste
        cost = (
            x[self.id[(0,0)]] + x[self.id[(0,1)]] + x[self.id[(0,2)]] +
            2.0*(x[self.id[(1,0)]] + x[self.id[(1,1)]] + x[self.id[(1,2)]]) +
            1.2*(x[self.id[(2,0)]] + x[self.id[(2,1)]] + x[self.id[(2,2)]])
        )
        out["F"] = cost

        # Sums por sink y source
        flow_to_sink   = np.array([
            x[self.id[(0,0)]] + x[self.id[(1,0)]] + x[self.id[(2,0)]],
            x[self.id[(0,1)]] + x[self.id[(1,1)]] + x[self.id[(2,1)]],
            x[self.id[(0,2)]] + x[self.id[(1,2)]] + x[self.id[(2,2)]],
        ])
        flow_from_src  = np.array([
            x[self.id[(0,0)]] + x[self.id[(0,1)]] + x[self.id[(0,2)]],
            x[self.id[(1,0)]] + x[self.id[(1,1)]] + x[self.id[(1,2)]],
            x[self.id[(2,0)]] + x[self.id[(2,1)]] + x[self.id[(2,2)]],
        ])

        # -------- 3) Restricciones G ≤ 0 ----------
        eps = 1e-8                      # tolerancia para (quasi)-igualdades
        g_eq = np.abs(flow_to_sink - self.demand) - eps   # 3 “igualdades”
        g_ineq = flow_from_src - self.supply              # 3 “≤”
        out["G"] = np.concatenate([g_eq, g_ineq])


# ---------- 4) Algoritmo genético ----------
algorithm = GA(pop_size=1000,
               repair=TransportRepair(max_iter=1000, tol=1e-12),   # o ras_adjust
    crossover=SBX(prob=0.9, eta=5),
    mutation=PolynomialMutation(eta=25),
               eliminate_duplicates=True)
#algorithm = GA(
#    pop_size=3000,              # p.ej. 300 individuos
#    eliminate_duplicates=True
#)

# Parada: 10 000 evaluaciones o antes si converge
termination = get_termination("n_eval", 50000)

# ---------- 5) Optimización ----------
res = minimize(
    problem     = TransportProblem(),
    algorithm   = algorithm,
    termination = termination,
    seed        = 42,
    save_history=True,
    verbose=True
)

# ---------- 6) Resultados ----------
print(f"Coste óptimo ≈ {res.F[0]:.4f}")
print("Flujos óptimos (matriz 3×3):")
X = res.X.reshape(3, 3)
print(X)
print("Demanda atendida:", X.sum(axis=0))
print("Oferta utilizada:", X.sum(axis=1))

# Visualización del coste vs generación (opcional)
hist_cost = [a.pop.get("F").min() for a in res.history]
points = np.column_stack([np.arange(len(hist_cost)), hist_cost])
Scatter(title="Histórico del mínimo por generación") \
    .add(points) \
    .show()