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


# ── Piece 6: resection (place a new camera from known 3D points) ─────────────

def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _rodrigues(w):
    """Turn a rotation vector w (axis*angle) into a 3x3 rotation matrix."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = _skew(w / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def refine_pose(R, t, X, x, K, iters=100):
    """
    Nudge (R, t) to minimize the reprojection error - the DLT resection gives a
    good starting guess, this polishes it. Rotation updates ride on the manifold
    via a small rotation vector (R <- expm[w] R) so R stays a valid rotation;
    translation is updated directly. Uses Levenberg-Marquardt: a damping term
    keeps steps sane and any step that does NOT reduce the error is rejected (so
    a bad start cannot make it diverge). Same downhill-least-squares idea Step 7
    later applies to all cameras at once.
    """
    def residuals(R, t):
        xh = (K @ (R @ X.T + t[:, None])).T
        return (xh[:, :2] / xh[:, 2:] - x).ravel()

    r0 = residuals(R, t)
    cost = r0 @ r0
    lam = 1e-3
    for _ in range(iters):
        J = np.zeros((len(r0), 6))
        eps = 1e-6
        for i in range(3):                          # 3 rotation columns
            dw = np.zeros(3); dw[i] = eps
            J[:, i] = (residuals(_rodrigues(dw) @ R, t) - r0) / eps
        for i in range(3):                          # 3 translation columns
            dt = np.zeros(3); dt[i] = eps
            J[:, 3 + i] = (residuals(R, t + dt) - r0) / eps
        H = J.T @ J
        g = J.T @ r0
        stepped = False
        for _try in range(12):                      # grow damping until a step helps
            try:
                delta = np.linalg.solve(H + lam * np.diag(np.diag(H)), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            Rn = _rodrigues(delta[:3]) @ R
            tn = t + delta[3:]
            rn = residuals(Rn, tn)
            cn = rn @ rn
            if cn < cost:                           # accept only if it improves
                R, t, r0, cost = Rn, tn, rn, cn
                lam = max(lam * 0.5, 1e-9)
                stepped = True
                break
            lam *= 10
        if not stepped or np.linalg.norm(delta) < 1e-10:
            break
    return R, t


def translation_given_rotation(R, X, x, K):
    """
    Linear least-squares for a camera's translation t when its rotation R is
    already known (e.g. R came from a two-view essential matrix). With R fixed,
    the projection  x ~ K (R X + t)  is linear in t, giving two equations per
    point. This stays well-conditioned even for near-planar points, because the
    unstable part (rotation) is no longer being solved for.
    """
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    a = (R @ X.T).T                                # R X, known per point
    A, b = [], []
    for (u, v), ai in zip(x, a):
        A.append([f, 0, -(u - cx)]); b.append((u - cx) * ai[2] - f * ai[0])
        A.append([0, f, -(v - cy)]); b.append((v - cy) * ai[2] - f * ai[1])
    t, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return t


def resect_camera(X, x, K):
    """
    Solve for a camera's pose (R, t) from >= 6 correspondences between known
    3D points X (N,3) and their pixels x (N,2), given the camera settings K.
    This is the mirror image of triangulation: there the cameras were known and
    the point unknown; here the points are known and the camera is unknown.

    Method (normalized DLT):
      1. Turn pixels into normalized rays  xn = K^-1 [u, v, 1]  (removes K).
      2. Normalize the 3D points (centre + isotropic scale) for numerical
         stability, exactly as _normalize does in 2D but in 3D.
      3. Each point gives two linear rows in the 12 entries of M = [R | t]
         (same "clear the ~ division" trick as triangulate); solve M m = 0 with
         SVD -> a 3x4 matrix known up to scale and sign, then undo step 2.
      4. Fix the sign so the rotation block is a proper rotation (det > 0),
         which is also the sign that puts points in front of the camera.
      5. Snap the rotation block to an exact rotation (SVD -> U V^T) and read
         the scale off its singular values to recover the true t.
    Returns (R, t).
    """
    Kinv = np.linalg.inv(K)
    xn = (Kinv @ np.c_[x, np.ones(len(x))].T).T
    xn = xn[:, :2] / xn[:, 2:]                     # normalized image coords

    # isotropic 3D normalization: centre at origin, mean distance sqrt(3)
    c = X.mean(axis=0)
    d = np.sqrt(((X - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(3) / d
    T = np.array([[s, 0, 0, -s * c[0]],
                  [0, s, 0, -s * c[1]],
                  [0, 0, s, -s * c[2]],
                  [0, 0, 0, 1.0]])
    Xn = (T @ np.c_[X, np.ones(len(X))].T).T       # normalized homogeneous pts

    rows = []
    for (un, vn), Xi in zip(xn, Xn):
        z = np.zeros(4)
        rows.append(np.r_[-Xi, z, un * Xi])        # un*(M3.X) - (M1.X) = 0
        rows.append(np.r_[z, -Xi, vn * Xi])        # vn*(M3.X) - (M2.X) = 0
    _, _, Vt = np.linalg.svd(np.array(rows))
    M = Vt[-1].reshape(3, 4) @ T                   # undo the 3D normalization

    if np.linalg.det(M[:, :3]) < 0:                # pick the physical sign
        M = -M

    U, D, Vt2 = np.linalg.svd(M[:, :3])            # nearest exact rotation
    R = U @ Vt2
    t = M[:, 3] / D.mean()                          # undo the leftover scale

    # DLT is only an initial guess; polish it to minimize reprojection error
    R, t = refine_pose(R, t, X, x, K)
    return R, t
