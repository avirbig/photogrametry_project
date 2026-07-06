"""
geometry.py
Hand-rolled multi-view geometry for the photogrammetry project.

Everything here is built from basic linear algebra (numpy for matrix
arithmetic and SVD only) - no computer-vision library is used. Each function
corresponds to one of the mathematical "pieces" explained in the logbook.

Conventions
-----------
  - A pixel is (x, y): x = column (from left), y = row (from top).
  - "Homogeneous" pixel = (x, y, 1); homogeneous 3D point = (X, Y, Z, 1).
  - A camera is a 3x4 projection matrix  P = K [ R | t ].
      R = 3x3 rotation (which way the camera points)
      t = 3-vector tied to the camera position
      K = 3x3 intrinsic matrix (focal length + image centre), see build_K.
  - Camera 1 (the reference) is always P1 = K1 [ I | 0 ].
"""
from __future__ import annotations
import numpy as np


# ── intrinsics ──────────────────────────────────────────────────────────────

def build_K(f: float, cx: float, cy: float) -> np.ndarray:
    """The camera settings matrix K (Step 3): focal f, image centre (cx, cy)."""
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1.0]])


# ── Piece 2/3: the fundamental matrix via the normalized 8-point algorithm ───

def _normalize(pts: np.ndarray):
    """
    Hartley normalization: shift points to have their centre at the origin and
    scale them so their average distance from the origin is sqrt(2). This keeps
    the 8-point algorithm numerically stable. Returns (normalized_h, T) where
    T is the 3x3 transform applied, so original_h = inv(T) @ normalized_h.
    """
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1.0]])
    pts_h = np.c_[pts, np.ones(len(pts))]
    return (T @ pts_h.T).T, T


def fundamental_from_points(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Estimate the fundamental matrix F from matched pixels so that
    x2^T F x1 = 0 for every correspondence. x1, x2 are (N,2) pixel arrays.

    Steps:
      1. Normalize both point sets (numerical stability).
      2. Build the (N,9) matrix A, one row per correspondence.
      3. Solve A f = 0 with SVD (f = smallest-singular-value direction).
      4. Enforce rank 2 (force F's smallest singular value to 0).
      5. Undo the normalization.
    """
    n1, T1 = _normalize(x1)
    n2, T2 = _normalize(x2)
    u1, v1 = n1[:, 0], n1[:, 1]
    u2, v2 = n2[:, 0], n2[:, 1]

    A = np.c_[u2 * u1, u2 * v1, u2,
              v2 * u1, v2 * v1, v2,
              u1,      v1,      np.ones(len(u1))]

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)          # smallest singular vector -> F draft

    # enforce rank 2
    U, S, Vt2 = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt2

    # undo normalization: x2^T F x1 = 0  ->  F_pixels = T2^T F_norm T1
    F = T2.T @ F @ T1
    return F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F


def epipolar_residuals(F, x1, x2) -> np.ndarray:
    """|x2^T F x1| for each pair - how far each match is from satisfying F."""
    x1h = np.c_[x1, np.ones(len(x1))]
    x2h = np.c_[x2, np.ones(len(x2))]
    return np.abs(np.sum((x2h @ F) * x1h, axis=1))


# ── Piece 4/5: essential matrix and pose (R, t) ──────────────────────────────

def essential_from_fundamental(F, K1, K2) -> np.ndarray:
    """
    E = K2^T F K1, then forced to a valid essential matrix (singular values
    exactly (1, 1, 0)). E holds the pure relative rotation+translation with the
    camera settings removed.
    """
    E = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E)
    return U @ np.diag([1, 1, 0]) @ Vt


def decompose_essential(E):
    """
    Split E into its two possible rotations (R_a, R_b) and a translation
    direction t. Combined they give four candidate poses; the caller picks the
    physically correct one with the cheirality check.
    """
    U, _, Vt = np.linalg.svd(E)
    # make U and Vt proper rotations (determinant +1) so R comes out valid
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1.0]])
    R_a = U @ W @ Vt
    R_b = U @ W.T @ Vt
    t = U[:, 2]
    return R_a, R_b, t


# ── Piece 5 helper: triangulation and the cheirality choice ──────────────────

def triangulate(P1, P2, x1, x2) -> np.ndarray:
    """
    For each matched pair, find the 3D point whose projections best match both
    clicks (the closest crossing of the two sightlines). Linear DLT method:
    build a 4x4 system per point and take its smallest-singular-value solution.
    Returns (N,3) array of 3D points.
    """
    X = []
    for a, b in zip(x1, x2):
        A = np.array([a[0] * P1[2] - P1[0],
                      a[1] * P1[2] - P1[1],
                      b[0] * P2[2] - P2[0],
                      b[1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X.append(Xh[:3] / Xh[3])
    return np.array(X)


def select_pose(R_a, R_b, t, K1, K2, x1, x2):
    """
    Try all four (R, t) candidates and keep the one that places the most points
    IN FRONT of both cameras (positive depth) - the cheirality check. Returns
    (R, t, X, n_in_front, counts) where X is the triangulated cloud for the
    winner and counts is the per-candidate in-front tally.
    """
    P1 = K1 @ np.eye(3, 4)
    candidates = [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]
    counts = []
    results = []
    for R, tt in candidates:
        P2 = K2 @ np.c_[R, tt]
        X = triangulate(P1, P2, x1, x2)
        depth1 = X[:, 2]
        depth2 = (R @ X.T + tt[:, None]).T[:, 2]
        n = int(((depth1 > 0) & (depth2 > 0)).sum())
        counts.append(n)
        results.append((R, tt, X, n))
    best = int(np.argmax(counts))
    R, tt, X, n = results[best]
    return R, tt, X, n, counts


def reprojection_error(P, X, x) -> np.ndarray:
    """Pixel distance between each projected 3D point and its clicked pixel."""
    xh = (P @ np.c_[X, np.ones(len(X))].T).T
    xp = xh[:, :2] / xh[:, 2:]
    return np.sqrt(((xp - x) ** 2).sum(axis=1))
