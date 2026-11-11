"""Contrast: a solution with limited regularity.
    u(x,y) = x^alpha sin(pi y),  alpha = 1.25   ->  u in H^s for s < alpha + 1/2 = 1.75
    f = -Δu = (pi^2 x^alpha - alpha (alpha-1) x^(alpha-2)) sin(pi y)      (singular at x = 0)
Theory: the rate is limited by the regularity of u, so P2 loses its advantage over P1.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fem2d import *

al = 1.25
u_ex = lambda x, y: x**al * np.sin(np.pi * y)
grad = lambda x, y: (al * x**(al - 1) * np.sin(np.pi * y), np.pi * x**al * np.cos(np.pi * y))
f = lambda x, y: (np.pi**2 * x**al - al * (al - 1) * x**(al - 2)) * np.sin(np.pi * y)

Ns = [4, 8, 16, 32, 64, 128]
res = {1: [], 2: []}
print(f"{'P':>2} {'N':>4} {'L2 error':>12} {'rate':>6} {'H1 error':>12} {'rate':>6}")
for degree in (1, 2):
    prev = None
    for N in Ns:
        V = FESpace(*unit_square_mesh(N), degree, nquad=10)
        uh = solve_poisson(V, f, u_ex)
        l2, h1 = V.errors(uh, u_ex, grad)
        r2, r1 = (np.log2(prev[0] / l2), np.log2(prev[1] / h1)) if prev else (float("nan"),) * 2
        prev = (l2, h1); res[degree].append((1 / N, l2, h1))
        print(f"{degree:>2} {N:>4} {l2:12.3e} {r2:6.2f} {h1:12.3e} {r1:6.2f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, name in enumerate(["L2 error", "H1-seminorm error"]):
    for d, mk in ((1, "o-"), (2, "s-")):
        a = np.array(res[d]); ax[k].loglog(a[:, 0], a[:, k + 1], mk, label=f"P{d}")
    ax[k].set_xlabel("mesh size h"); ax[k].set_ylabel(name); ax[k].grid(True, which="both", alpha=0.3); ax[k].legend()
fig.suptitle("Limited regularity u = x^1.25 sin(pi y): P2 no longer beats P1 by much")
fig.tight_layout(); fig.savefig("convergence_nonsmooth.png", dpi=150)
