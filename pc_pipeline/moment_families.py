"""Discrete orthogonal moment families (Phase 7): Tchebichef, Krawtchouk, Hahn.

All three are orthonormal polynomial bases on N points with respect to a weight w(x):
    Tchebichef  w(x) = 1                                   (global, uniform)
    Krawtchouk  w(x) = C(N-1, x) p^x (1-p)^(N-1-x)          (local, centred at p (N-1))
    Hahn        w(x) = C(a+x, x) C(b+N-1-x, N-1-x)          (between: a=b=0 is Tchebichef)
The weighted basis column n is sqrt(w(x)) P_n(x), with P_n orthonormal under w; it is built
with the Stieltjes procedure (three-term recurrence from the weight) plus two passes of full
re-orthogonalisation, which is stable in float64 for discrete weights. The uniform weight
reproduces gtao_grid.tchebichef_basis exactly (checked by self_check()), so every family
comes from the same code path. Lower orders are the leading columns, as for Tchebichef.
"""
from functools import lru_cache
from math import lgamma, log

import numpy as np


def log_weight(kind, N, a=0.5, b=0.0):
    x = np.arange(N, dtype=np.float64)
    if kind == "tcheb":
        return np.zeros(N)
    if kind == "kraw":                       # a = p
        n = N - 1
        return np.array([lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
                         + k * log(a) + (n - k) * log(1 - a) for k in x])
    if kind == "hahn":                       # a = alpha, b = beta
        n = N - 1
        return np.array([lgamma(a + k + 1) - lgamma(k + 1) - lgamma(a + 1)
                         + lgamma(b + n - k + 1) - lgamma(n - k + 1) - lgamma(b + 1) for k in x])
    raise ValueError(kind)


@lru_cache(maxsize=None)
def basis(kind, N, K, a=0.5, b=0.0):
    """(N, K) float32 orthonormal weighted basis of the given family."""
    lw = log_weight(kind, N, a, b)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    s = np.sqrt(w)
    t = np.linspace(-1.0, 1.0, N)            # positive affine map of x: same polynomials
    Q = np.zeros((N, K))                     # Q[:, n] = sqrt(w) * P_n
    q = s.copy()
    for n in range(K):
        for _ in range(2):                   # full re-orthogonalisation, twice
            q -= Q[:, :n] @ (Q[:, :n].T @ q)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            raise ValueError(f"{kind} basis degenerate at order {n} (N={N}, a={a}, b={b})")
        Q[:, n] = q / norm
        q = t * Q[:, n]                      # next degree: x * P_n, then orthogonalise
    return Q.astype(np.float32)


# Face-crop families (rows = y from top, columns = x from left). Eyes sit at ~0.38 and the
# mouth at ~0.72 of the face-box height (checked on a DIPSER crop, 2026-11-20).
FAMILIES = {
    "tcheb": (("tcheb", 0.5, 0.0), ("tcheb", 0.5, 0.0)),
    "hahn": (("hahn", 4.0, 4.0), ("hahn", 4.0, 4.0)),
    "kraw": (("kraw", 0.5, 0.0), ("kraw", 0.5, 0.0)),
    "kraw_eyes": (("kraw", 0.38, 0.0), ("kraw", 0.5, 0.0)),
    "kraw_mouth": (("kraw", 0.72, 0.0), ("kraw", 0.5, 0.0)),
}


def moments_2d(img, family, K):
    """(N, N) image -> (K, K) moments: B_y.T @ img @ B_x."""
    (ky, ay, by), (kx, ax, bx) = FAMILIES[family]
    By = basis(ky, img.shape[0], K, ay, by)
    Bx = basis(kx, img.shape[1], K, ax, bx)
    return (By.T @ img) @ Bx


def self_check(grid=None, N=64, K=16):
    """Orthonormality of every family, and uniform weight == gtao_grid Tchebichef."""
    report = {}
    for kind, a, b in (("tcheb", 0.5, 0.0), ("hahn", 4.0, 4.0), ("kraw", 0.5, 0.0),
                       ("kraw", 0.38, 0.0), ("kraw", 0.72, 0.0)):
        B = basis(kind, N, K, a, b).astype(np.float64)
        report[f"{kind}_{a}_{b}_orth_err"] = float(np.abs(B.T @ B - np.eye(K)).max())
    if grid is not None:
        for n in (N, 8, 12, 32):
            k = min(K, n)
            ref = grid.tchebichef_basis(n, k)
            report[f"tcheb_vs_grid_N{n}"] = float(np.abs(basis("tcheb", n, k) - ref).max())
    report["passed"] = all(v < 1e-4 for v in report.values())
    return report
