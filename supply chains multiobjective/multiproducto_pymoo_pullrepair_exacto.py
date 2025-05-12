import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
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


from scipy.optimize import minimize

class PullRepair(Repair):
    def __init__(self, tol=1e-8):
        super().__init__()
        self.tol = tol

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X)  # ← esto es crítico para evitar errores con objetos de pymoo

        for n in range(len(X)):
            x_full = np.zeros((I, J, K))

            for j in range(J):
                comp = np.append(P[j], 1 - np.sum(P[j]))
                comp = comp / comp.sum()

                def get_index(i, k):
                    return i * K + k

                def objective(x):
                    flows = x.reshape((I, K))
                    total = np.sum(flows)
                    if total <= 0:
                        return 1e9
                    mix = flows.sum(axis=0) / total
                    return np.sum((mix - comp) ** 2)

                x0 = np.full(I * K, 1.0)
                bounds = [(0, D[i, k]) for i in range(I) for k in range(K)]

                A_eq = []
                b_eq = []
                for k in range(K - 1):
                    row = np.zeros(I * K)
                    for i in range(I):
                        row[get_index(i, k)] = 1.0
                    for i in range(I):
                        for m in range(K):
                            row[get_index(i, m)] -= comp[k]
                    A_eq.append(row)
                    b_eq.append(0.0)

                constraints = [{'type': 'eq', 'fun': lambda x, A=A_eq[i], b=b_eq[i]: A @ x - b} for i in range(len(A_eq))]

                result = minimize(
                    objective,
                    x0,
                    bounds=bounds,
                    constraints=constraints,
                    method='SLSQP',
                    options={'ftol': 1e-9, 'disp': False}
                )

                if result.success:
                    x_sol = result.x.reshape((I, K))
                    x_full[:, j, :] = x_sol
                else:
                    x_full[:, j, :] = 0.0  # fallback si no converge

            # Clipping final a los límites del problema
            x_full = np.clip(x_full, problem.xl.reshape((I, J, K)), problem.xu.reshape((I, J, K)))
            X[n] = x_full.ravel()

        return X


class MultiproductTransportProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=I*J*K, n_obj=1, n_constr=0,
                         xl=np.zeros(I*J*K), xu=np.full(I*J*K, 1e3))
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -np.sum(x)

problem = MultiproductTransportProblem()

algorithm = GA(
    pop_size=5000,
    repair=PullRepair(),
    crossover=SBX(prob=0.9, eta=5),
    mutation=PolynomialMutation(eta=5),
    eliminate_duplicates=True
)

termination = get_termination("n_eval", 100000)

res = minimize(
    problem=problem,
    algorithm=algorithm,
    termination=termination,
    seed=21,
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
