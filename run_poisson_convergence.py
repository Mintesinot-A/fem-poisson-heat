"""Spatial convergence study for -Δu = f on the unit square with a smooth manufactured solution
    u(x,y) = exp(x) sin(pi y)      (non-zero Dirichlet data on the boundary)
    f = -Δu = (pi^2 - 1) exp(x) sin(pi y)
Expected rates for smooth u:   P1: L2 ~ h^2, H1 ~ h      P2: L2 ~ h^3, H1 ~ h^2
"""
import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fem2d import *

u_ex = lambda x, y: np.exp(x) * np.sin(np.pi * y)
grad = lambda x, y: (np.exp(x) * np.sin(np.pi * y), np.pi * np.exp(x) * np.cos(np.pi * y))
f = lambda x, y: (np.pi**2 - 1) * np.exp(x) * np.sin(np.pi * y)

Ns = [4, 8, 16, 32, 64, 128]
rows, res = [], {}
for degree in (1, 2):
    prev = None
    for N in Ns:
        nodes, tri = unit_square_mesh(N)
        V = FESpace(nodes, tri, degree)
        uh = solve_poisson(V, f, u_ex)
        l2, h1 = V.errors(uh, u_ex, grad)
        r2 = r1 = float("nan")
        if prev:
            r2, r1 = np.log2(prev[0] / l2), np.log2(prev[1] / h1)
        prev = (l2, h1)
        rows.append([degree, N, 1 / N, V.ndof, l2, r2, h1, r1])
        res.setdefault(degree, []).append((1 / N, l2, h1))

with open("results_poisson.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["degree", "N", "h", "dofs", "L2_error", "L2_rate", "H1_error", "H1_rate"])
    w.writerows(rows)

print(f"{'P':>2} {'N':>4} {'dofs':>8} {'L2 error':>12} {'rate':>6} {'H1 error':>12} {'rate':>6}")
for d, N, h, nd, l2, r2, h1, r1 in rows:
    print(f"{d:>2} {N:>4} {nd:>8} {l2:12.3e} {r2:6.2f} {h1:12.3e} {r1:6.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (name, col) in enumerate([("L2 error", 1), ("H1-seminorm error", 2)]):
    for d, mk in ((1, "o-"), (2, "s-")):
        a = np.array(res[d])
        ax[k].loglog(a[:, 0], a[:, col], mk, label=f"P{d}")
    h = np.array([1 / n for n in Ns])
    slopes = {0: (2, 3), 1: (1, 2)}[k]
    for d, s in zip((1, 2), slopes):
        a = np.array(res[d])
        ax[k].loglog(h, a[0, col] * (h / h[0]) ** s, "k--", lw=0.8)
        ax[k].text(h[-2], a[0, col] * (h[-2] / h[0]) ** s * 1.6, f"slope {s}", fontsize=8)
    ax[k].set_xlabel("mesh size h"); ax[k].set_ylabel(name); ax[k].grid(True, which="both", alpha=0.3); ax[k].legend()
fig.suptitle("Poisson, smooth solution u = exp(x) sin(pi y): convergence of Galerkin FEM")
fig.tight_layout(); fig.savefig("convergence_poisson.png", dpi=150)
