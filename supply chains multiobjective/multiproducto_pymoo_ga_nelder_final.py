import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.algorithms.combination.local_single_objective import LocalSingleObjective
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
import matplotlib.pyplot as plt

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

T_min = [1.0, 1.0, 1.0]

dmap = {}
k = 0
for i in range(I):
    for j in range(J):
        for l in range(K):
            dmap[k] = (i, j, l)
            k += 1
idmap = {v: k for k, v in dmap.items()}

class FullRepair(Repair):
    def __init__(self, max_iter=1000, tol=1e-12):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def _do(self, problem, X, **kwargs):
        for n in range(len(X)):
            x = X[n].copy().reshape((I, J, K))

            for j in range(J):
                total = x[:, j, :].sum()
                total = max(total, T_min[j])
                comp = np.append(P[j], 1 - np.sum(P[j]))
                x[:, j, :] = 0.0
                for i in range(I):
                    max_prod = D[i]
                    max_frac = max_prod / np.maximum(1, max_prod.sum())
                    x[i, j, :] = np.minimum(max_prod, comp * total * max_frac)

            for i in range(I):
                for k in range(K):
                    flujo = x[i, :, k].sum()
                    if flujo > D[i, k]:
                        x[i, :, k] *= D[i, k] / flujo

            x = np.clip(x, problem.xl.reshape((I, J, K)), problem.xu.reshape((I, J, K)))
            X[n] = x.ravel()
        return X

class MultiproductTransportProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=I*J*K, n_obj=1, n_constr=0,
                         xl=np.zeros(I*J*K), xu=np.full(I*J*K, 1e3))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -np.sum(x)

problem = MultiproductTransportProblem()

ga = GA(
    pop_size=1000,
    repair=FullRepair(),
    crossover=SBX(prob=0.9, eta=5),
    mutation=PolynomialMutation(eta=25),
    eliminate_duplicates=True
)

nelder = NelderMead()
algorithm = LocalSingleObjective(global_opt=ga, local=nelder, weight=0.9)
termination = get_termination("n_eval", 50000)

res = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=termination,
    seed=42,
    verbose=True,
    save_history=True
)

if res.F is not None:
    print(f"\n✅ Flujo total máximo: {-res.F[0]:.4f}")
    x_opt = res.X
    for k, (i, j, l) in dmap.items():
        if x_opt[k] > 1e-6:
            print(f"Source {i} → Sink {j}, producto {l}: {x_opt[k]:.2f}")

    x_tensor = res.X.reshape((I, J, K))
    print("\nVerificación de proporciones por destino:")
    for j in range(J):
        sum_product = x_tensor[:, j, :].sum(axis=0)
        total = sum_product.sum()
        if total > 0:
            porcentajes = sum_product / total
            esperado = np.append(P[j], 1 - np.sum(P[j]))
            print(f"Sink {j}: Total = {total:.2f} | Proporciones = {porcentajes.round(4)} | Esperado = {esperado}")
        else:
            print(f"Sink {j}: No recibió producto.")

    # Gráfico histórico
    if hasattr(res, "history"):
        min_hist = [a.opt.get("F")[0] for a in res.history if a.opt is not None]
        plt.plot(np.arange(len(min_hist)), -np.array(min_hist), marker='o')
        plt.xlabel("Generación")
        plt.ylabel("Mejor valor de flujo total")
        plt.title("Evolución del valor óptimo")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

else:
    print("\n❌ Optimización fallida: No se encontró solución factible.")
