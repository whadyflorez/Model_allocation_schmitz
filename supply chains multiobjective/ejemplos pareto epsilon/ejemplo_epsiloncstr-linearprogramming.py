import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ---------- Datos del problema (LP) ----------
# Objetivos (minimizar)
c1 = np.array([3.0, 1.0])   # f1(x) = 3 x1 +  x2
c2 = np.array([1.0, 2.0])   # f2(x) = 1 x1 + 2 x2

# Restricciones Ax <= b; x >= 0
A = np.array([
    [2.0, 1.0],  # 2x1 + x2  <= 14
    [1.0, 2.0],  #  x1 + 2x2 <= 10
    [1.0, 0.0],  #  x1       <= 6
    [0.0, 1.0],  #       x2  <= 6
])
b = np.array([14.0, 10.0, 6.0, 6.0])

# ---------- Utilidades ----------
def solve_lp_minimize(c, A, b, method='highs-ds'):
    """Resuelve min c^T x s.a. Ax<=b, x>=0 con simplex (HiGHS dual simplex)."""
    res = linprog(c=c, A_ub=A, b_ub=b, bounds=[(0, None)]*A.shape[1], method=method)
    if not res.success:
        return None
    x = res.x
    return {
        "x": x,
        "f": c @ x,
        "status": res.status,
        "message": res.message,
    }

def augment_with_epsilon_constraint(A, b, c2, eps):
    """Agrega la restricción f2(x) = c2^T x <= eps al sistema Ax<=b."""
    A_aug = np.vstack([A, c2])
    b_aug = np.hstack([b, eps])
    return A_aug, b_aug

# ---------- Extremos para determinar el rango de epsilons ----------
# (1) Mejor f2 (minimiza f2)
ext_f2 = solve_lp_minimize(c2, A, b)
if ext_f2 is None:
    raise RuntimeError("El LP para minimizar f2 es infactible.")

f2_min = ext_f2["f"]

# (2) Mejor f1 (minimiza f1) y el f2 resultante en ese punto
ext_f1 = solve_lp_minimize(c1, A, b)
if ext_f1 is None:
    raise RuntimeError("El LP para minimizar f1 es infactible.")

f2_at_f1_opt = c2 @ ext_f1["x"]

# Asegura un rango creciente de eps para barrer el frente
eps_start = f2_min
eps_end   = max(f2_min, f2_at_f1_opt)
if np.isclose(eps_start, eps_end):
    # Si coinciden (raro, pero posible), expandimos un poco el extremo superior
    eps_end = eps_start + 5.0

# ---------- Barrido ε-constraint ----------
n_eps = 25
eps_values = np.linspace(eps_start, eps_end, n_eps)

pareto = []
for eps in eps_values:
    A_eps, b_eps = augment_with_epsilon_constraint(A, b, c2, eps)
    sol = solve_lp_minimize(c1, A_eps, b_eps)
    if sol is None:
        # infactible para este epsilon; lo omitimos
        continue
    x = sol["x"]
    f1_val = c1 @ x
    f2_val = c2 @ x
    pareto.append((f1_val, f2_val, x))

# Elimina duplicados numéricos cercanos (opcional)
def unique_points(points, tol=1e-6):
    uniq = []
    for p in points:
        if not any(np.allclose(p[:2], q[:2], atol=tol, rtol=0) for q in uniq):
            uniq.append(p)
    return uniq

pareto = unique_points(pareto)

# Ordenamos por f2 (o por f1)
pareto.sort(key=lambda t: t[1])

# ---------- Resultados ----------
print("Extremo (min f2):")
print("  x* =", ext_f2["x"], "  f1 =", c1 @ ext_f2["x"], "  f2 =", ext_f2["f"])
print("Extremo (min f1):")
print("  x* =", ext_f1["x"], "  f1 =", ext_f1["f"], "  f2 =", f2_at_f1_opt)

print("\nPuntos Pareto aproximados (ε-constraint):")
for f1_val, f2_val, x in pareto:
    print(f"  f1={f1_val:8.4f}  f2={f2_val:8.4f}   x={x}")

# ---------- Visualización ----------
f1_vals = [p[0] for p in pareto]
f2_vals = [p[1] for p in pareto]

plt.figure()
plt.plot(f2_vals, f1_vals, marker='o')
plt.xlabel(r'$f_2(x)=x_1+2x_2$')
plt.ylabel(r'$f_1(x)=3x_1+x_2$')
plt.title('Frente de Pareto (método ε-constraint + HiGHS Dual Simplex)')
plt.grid(True)
plt.show()
