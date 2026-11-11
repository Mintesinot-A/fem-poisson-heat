"""Heat equation u_t - Δu = f on the unit square, manufactured smooth solution
    u(x,y,t) = exp(-t) exp(x) sin(pi y),   f = (pi^2 - 2) exp(-t) exp(x) sin(pi y)
Time stepping: theta-method (theta=1: backward Euler, order 1; theta=1/2: Crank-Nicolson, order 2).
Spatial discretisation: P2 elements on a fine mesh, so that the time error dominates.
"""
import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fem2d import *

phi_ = lambda x, y: np.exp(x) * np.sin(np.pi * y)
u_ex = lambda x, y, t: np.exp(-t) * phi_(x, y)
f = lambda x, y, t: (np.pi**2 - 2) * np.exp(-t) * phi_(x, y)
grad = lambda x, y, t: (np.exp(-t) * np.exp(x) * np.sin(np.pi * y), np.exp(-t) * np.pi * np.exp(x) * np.cos(np.pi * y))

T, N = 0.5, 48
nodes, tri = unit_square_mesh(N)
V = FESpace(nodes, tri, 2)
steps = [4, 8, 16, 32, 64, 128]

# reference: the same spatial discretisation with a very small time step (isolates the time error)
ref = solve_heat(V, f, u_ex, lambda x, y: u_ex(x, y, 0.0), T, 2048, theta=0.5)
l2_ref = lambda u: np.sqrt((u - ref) @ (V.mass @ (u - ref)))
# total error (time + space) against the exact solution
l2_ex, _ = V.errors(ref, lambda x, y: u_ex(x, y, T), lambda x, y: grad(x, y, T))
print(f"Spatial error floor (P2, N={N}, tiny dt): {l2_ex:.3e}\n")

rows = []
out = {1.0: [], 0.5: []}
for theta in (1.0, 0.5):
    prev = None
    for ns in steps:
        u = solve_heat(V, f, u_ex, lambda x, y: u_ex(x, y, 0.0), T, ns, theta)
        e = l2_ref(u)
        rate = np.log2(prev / e) if prev else float("nan")
        prev = e
        rows.append([("BE" if theta == 1 else "CN"), ns, T / ns, e, rate])
        out[theta].append((T / ns, e))
with open("results_heat.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scheme", "steps", "dt", "L2_time_error", "rate"]); w.writerows(rows)
print(f"{'scheme':>6} {'steps':>6} {'dt':>9} {'time error (L2)':>16} {'rate':>6}")
for s, ns, dt, e, r in rows:
    print(f"{s:>6} {ns:>6} {dt:9.4f} {e:16.3e} {r:6.2f}")

fig, ax = plt.subplots(figsize=(5.5, 4))
for theta, lab, mk in ((1.0, "Backward Euler", "o-"), (0.5, "Crank-Nicolson", "s-")):
    a = np.array(out[theta]); ax.loglog(a[:, 0], a[:, 1], mk, label=lab)
dts = np.array([T / s for s in steps])
ax.loglog(dts, out[1.0][0][1] * (dts / dts[0]), "k--", lw=0.8); ax.text(dts[-2], out[1.0][0][1] * dts[-2] / dts[0] * 1.5, "slope 1", fontsize=8)
ax.loglog(dts, out[0.5][0][1] * (dts / dts[0]) ** 2, "k--", lw=0.8); ax.text(dts[-2], out[0.5][0][1] * (dts[-2] / dts[0]) ** 2 * 1.5, "slope 2", fontsize=8)
ax.set_xlabel("time step dt"); ax.set_ylabel("L2 time-discretisation error at T = 0.5")
ax.grid(True, which="both", alpha=0.3); ax.legend(); ax.set_title("Heat equation: temporal convergence (P2 in space)")
fig.tight_layout(); fig.savefig("convergence_heat.png", dpi=150)
