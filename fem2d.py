"""
fem2d.py -- a small, readable Galerkin finite element code in Python (NumPy/SciPy)
for scalar problems on the unit square with triangular P1 and P2 elements.

Model problems
    Poisson:  -Δu = f          in Ω = (0,1)^2,  u = g on ∂Ω
    Heat:      u_t - Δu = f     in Ω,            u = g on ∂Ω, u(0) = u0

Weak form (Poisson):  find u in V_g such that
    ∫ ∇u·∇v dx = ∫ f v dx          for all v in V_0 = H^1_0(Ω)
Galerkin: replace V by a finite element space V_h (piecewise P1 or P2).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# Mesh: structured triangulation of the unit square (each square -> 2 triangles)
# ----------------------------------------------------------------------------
def unit_square_mesh(N):
    xs = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.c_[X.ravel(), Y.ravel()]
    idx = lambda i, j: i * (N + 1) + j
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i, j = i.ravel(), j.ravel()
    n00, n10, n01, n11 = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
    tri = np.vstack([np.c_[n00, n10, n11], np.c_[n00, n11, n01]])  # counter-clockwise
    return nodes, tri


# ----------------------------------------------------------------------------
# Quadrature on the reference triangle (collapsed Gauss rule, exact to degree 2n-2)
# ----------------------------------------------------------------------------
def triangle_quadrature(n=6):
    s, w = np.polynomial.legendre.leggauss(n)
    s, w = (s + 1) / 2, w / 2
    xi = np.repeat(s, n)
    eta = (s[None, :] * (1 - s[:, None])).ravel()
    wt = (w[:, None] * w[None, :] * (1 - s[:, None])).ravel()
    return xi, eta, wt  # weights sum to 1/2 (area of the reference triangle)


# ----------------------------------------------------------------------------
# Reference basis functions (barycentric L0 = 1-xi-eta, L1 = xi, L2 = eta)
# ----------------------------------------------------------------------------
def reference_basis(degree, xi, eta):
    L = np.array([1 - xi - eta, xi, eta])                       # (3, nq)
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])       # (3, 2)
    if degree == 1:
        phi = L
        dphi = np.repeat(dL[:, :, None], len(xi), axis=2)
        return phi, dphi
    if degree == 2:
        phi = np.empty((6, len(xi)))
        dphi = np.empty((6, 2, len(xi)))
        for a in range(3):                                      # vertex functions
            phi[a] = L[a] * (2 * L[a] - 1)
            dphi[a] = (4 * L[a] - 1)[None, :] * dL[a][:, None]
        # edge functions: local dof 3 <-> edge (1,2), 4 <-> (2,0), 5 <-> (0,1)
        for k, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
            phi[3 + k] = 4 * L[a] * L[b]
            dphi[3 + k] = 4 * (L[b][None, :] * dL[a][:, None] + L[a][None, :] * dL[b][:, None])
        return phi, dphi
    raise ValueError("degree must be 1 or 2")


# ----------------------------------------------------------------------------
# Finite element space: dof numbering, geometry, assembly
# ----------------------------------------------------------------------------
class FESpace:
    def __init__(self, nodes, tri, degree=1, nquad=6):
        self.degree, self.tri = degree, tri
        nn = len(nodes)
        if degree == 1:
            self.dof_xy, self.elem_dofs = nodes, tri
        else:
            loc = [(1, 2), (2, 0), (0, 1)]
            pairs = np.stack([np.sort(tri[:, [a, b]], axis=1) for a, b in loc], axis=1)  # (nt,3,2)
            keys = pairs[..., 0] * nn + pairs[..., 1]
            uniq, inv = np.unique(keys.ravel(), return_inverse=True)
            a, b = uniq // nn, uniq % nn
            self.dof_xy = np.vstack([nodes, 0.5 * (nodes[a] + nodes[b])])
            self.elem_dofs = np.hstack([tri, nn + inv.reshape(-1, 3)])
        self.ndof = len(self.dof_xy)

        # geometry (affine triangles)
        x = nodes[tri]                                           # (nt,3,2)
        J = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]], axis=2)  # (nt,2,2)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        invJ = np.empty_like(J)
        invJ[:, 0, 0], invJ[:, 0, 1] = J[:, 1, 1] / det, -J[:, 0, 1] / det
        invJ[:, 1, 0], invJ[:, 1, 1] = -J[:, 1, 0] / det, J[:, 0, 0] / det
        xi, eta, wq = triangle_quadrature(nquad)
        self.phi, dphi = reference_basis(degree, xi, eta)        # (nb,nq), (nb,2,nq)
        # physical gradients: grad_x = J^{-T} grad_xi
        self.G = np.einsum("emk,bmq->ebkq", invJ, dphi)          # (nt,nb,2,nq)
        self.wdet = np.abs(det)[:, None] * wq[None, :]           # (nt,nq)
        self.xq = x[:, 0, :, None] + np.einsum("eij,jq->eiq", J, np.vstack([xi, eta]))  # (nt,2,nq)
        # boundary dofs (a dof is on the boundary if its coordinates are)
        X, Y = self.dof_xy[:, 0], self.dof_xy[:, 1]
        tol = 1e-12
        self.is_bnd = (X < tol) | (X > 1 - tol) | (Y < tol) | (Y > 1 - tol)
        self.h = 1.0 / np.sqrt(len(tri) / 2.0)                  # = 1/N (legs of the triangles)
        self._K = self._M = None

    # -- matrices --------------------------------------------------------
    def _assemble(self, Ke):
        nb = self.elem_dofs.shape[1]
        rows = np.repeat(self.elem_dofs, nb, axis=1).ravel()
        cols = np.tile(self.elem_dofs, (1, nb)).ravel()
        return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.ndof,) * 2).tocsr()

    @property
    def stiffness(self):          # K_ij = ∫ ∇φ_j · ∇φ_i
        if self._K is None:
            self._K = self._assemble(np.einsum("eq,ebkq,eckq->ebc", self.wdet, self.G, self.G))
        return self._K

    @property
    def mass(self):               # M_ij = ∫ φ_j φ_i
        if self._M is None:
            self._M = self._assemble(np.einsum("eq,bq,cq->ebc", self.wdet, self.phi, self.phi))
        return self._M

    def load(self, f):            # F_i = ∫ f φ_i ,  f(x, y) vectorised
        fq = f(self.xq[:, 0, :], self.xq[:, 1, :])
        Fe = np.einsum("eq,eq,bq->eb", self.wdet, fq, self.phi)
        return np.bincount(self.elem_dofs.ravel(), weights=Fe.ravel(), minlength=self.ndof)

    def interpolate(self, g):     # nodal interpolant
        return g(self.dof_xy[:, 0], self.dof_xy[:, 1])

    # -- error norms -----------------------------------------------------
    def errors(self, uh, u_ex, grad_ex):
        """Return (L2 error, H1-seminorm error) of uh against the exact solution."""
        ue = uh[self.elem_dofs]                                  # (nt,nb)
        uq = np.einsum("eb,bq->eq", ue, self.phi)
        gq = np.einsum("eb,ebkq->ekq", ue, self.G)
        ex = u_ex(self.xq[:, 0, :], self.xq[:, 1, :])
        gx, gy = grad_ex(self.xq[:, 0, :], self.xq[:, 1, :])
        l2 = np.sqrt(np.sum(self.wdet * (uq - ex) ** 2))
        h1 = np.sqrt(np.sum(self.wdet * ((gq[:, 0] - gx) ** 2 + (gq[:, 1] - gy) ** 2)))
        return l2, h1


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------
def solve_poisson(V, f, g):
    """-Δu = f, u = g on the boundary. Dirichlet values imposed by elimination."""
    K, F = V.stiffness, V.load(f)
    u = np.zeros(V.ndof)
    b = V.is_bnd
    u[b] = V.interpolate(g)[b]
    free = ~b
    rhs = F[free] - K[free][:, b] @ u[b]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u


def solve_heat(V, f, g, u0, T, nsteps, theta=1.0):
    """u_t - Δu = f. theta=1: backward Euler, theta=0.5: Crank-Nicolson.
       f(x, y, t), g(x, y, t) vectorised; u0(x, y)."""
    K, M = V.stiffness, V.mass
    dt = T / nsteps
    A = (M / dt + theta * K).tocsr()
    B = (M / dt - (1 - theta) * K).tocsr()
    b, free = V.is_bnd, ~V.is_bnd
    lu = spla.splu(A[free][:, free].tocsc())
    Abf = A[free][:, b]
    u = V.interpolate(u0)
    Fold = V.load(lambda x, y: f(x, y, 0.0))
    for n in range(nsteps):
        t1 = (n + 1) * dt
        Fnew = V.load(lambda x, y: f(x, y, t1))
        un = np.array(u)
        un[b] = V.interpolate(lambda x, y: g(x, y, t1))[b]
        rhs = B @ u + theta * Fnew + (1 - theta) * Fold
        un[free] = lu.solve(rhs[free] - Abf @ un[b])
        u, Fold = un, Fnew
    return u
