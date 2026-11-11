"""Sanity tests: (1) quadrature, (2) patch test, (3) stiffness of constants."""
import numpy as np
from fem2d import *

def test_quadrature():
    xi, eta, w = triangle_quadrature(6)
    assert abs(w.sum() - 0.5) < 1e-14
    # ∫ xi^3 eta^2 over reference triangle = 3! 2! / 7! = 1/420
    assert abs(np.sum(w * xi**3 * eta**2) - 1 / 420) < 1e-13

def test_patch(degree):
    """P1 and P2 must reproduce a linear solution exactly (u = 1 + 2x - 3y, f = 0)."""
    nodes, tri = unit_square_mesh(5)
    V = FESpace(nodes, tri, degree)
    u_ex = lambda x, y: 1 + 2 * x - 3 * y
    u = solve_poisson(V, lambda x, y: 0 * x, u_ex)
    err = np.max(np.abs(u - V.interpolate(u_ex)))
    assert err < 1e-12, err

def test_constants_in_kernel():
    nodes, tri = unit_square_mesh(4)
    V = FESpace(nodes, tri, 2)
    assert np.max(np.abs(V.stiffness @ np.ones(V.ndof))) < 1e-12
    assert abs(np.ones(V.ndof) @ (V.mass @ np.ones(V.ndof)) - 1.0) < 1e-12  # area of the square

if __name__ == "__main__":
    test_quadrature(); test_patch(1); test_patch(2); test_constants_in_kernel()
    print("All tests passed.")
