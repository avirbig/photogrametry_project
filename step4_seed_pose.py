"""
step4_seed_pose.py
Step 4 - Seed pair: recover the relative pose (rotation R and translation
direction t) of the second seed camera relative to the first, from their
shared marked points.

Seed pair: 01_front_clean.jpg + 04_front_elevated.jpg  (chosen because it is
non-planar and well-conditioned; see logbook Entry 8).

Because two views of a building do NOT reliably determine the focal length,
we FIX an initial focal by a normal-lens assumption (f = 1.2 * max(width,
height) pixels) and let bundle adjustment (Step 7) refine it later.

Run:
    .venv/Scripts/python.exe step4_seed_pose.py
"""
from __future__ import annotations
import json
import os
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import geometry as geo

ROOT = os.path.dirname(os.path.abspath(__file__))
CORR = os.path.join(ROOT, "data", "correspondences.json")
INTR = os.path.join(ROOT, "data", "camera_intrinsics.json")
OUT  = os.path.join(ROOT, "data", "seed_pose.json")

IMG_A = "01_front_clean.jpg"      # reference camera (origin)
IMG_B = "04_front_elevated.jpg"   # second camera


def assumed_focal(intr, img) -> float:
    """Normal-lens starting guess for focal length, in pixels."""
    return 1.2 * max(intr[img]["width"], intr[img]["height"])


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    marks = corr["marks"]

    # shared points
    shared = [(pid, marks[pid][IMG_A], marks[pid][IMG_B])
              for pid in corr["point_ids"]
              if IMG_A in marks.get(pid, {}) and IMG_B in marks.get(pid, {})]
    pids = [s[0] for s in shared]
    x1 = np.array([s[1] for s in shared], float)
    x2 = np.array([s[2] for s in shared], float)
    print(f"Seed pair : {IMG_A}  +  {IMG_B}")
    print(f"Shared marked points: {len(shared)}  ({', '.join(pids)})\n")

    # intrinsics (assumed focal)
    fA, fB = assumed_focal(intr, IMG_A), assumed_focal(intr, IMG_B)
    K1 = geo.build_K(fA, intr[IMG_A]["cx"], intr[IMG_A]["cy"])
    K2 = geo.build_K(fB, intr[IMG_B]["cx"], intr[IMG_B]["cy"])
    print(f"Assumed focal (normal-lens): fA={fA:.1f}px  fB={fB:.1f}px")
    print("K1 =\n", K1, "\nK2 =\n", K2, "\n")

    # Piece 2-3: fundamental matrix
    F = geo.fundamental_from_points(x1, x2)
    res = geo.epipolar_residuals(F, x1, x2)
    print("Fundamental matrix F =\n", F)
    print(f"epipolar residual |x2^T F x1|: mean={res.mean():.4f}px "
          f"max={res.max():.4f}px\n")

    # Piece 4: essential matrix
    E = geo.essential_from_fundamental(F, K1, K2)
    print("Essential matrix E =\n", E, "\n")

    # Piece 5: decompose + cheirality
    R_a, R_b, t = geo.decompose_essential(E)
    R, t, X, n_front, counts = geo.select_pose(R_a, R_b, t, K1, K2, x1, x2)
    print(f"cheirality (points in front of both cameras) for the 4 "
          f"candidates: {counts}")
    print(f"-> winner keeps {n_front}/{len(shared)} points in front\n")

    P1 = K1 @ np.eye(3, 4)
    P2 = K2 @ np.c_[R, t]
    e1 = geo.reprojection_error(P1, X, x1)
    e2 = geo.reprojection_error(P2, X, x2)
    ang = np.degrees(np.arccos((np.trace(R) - 1) / 2))

    print("RESULT")
    print("R (rotation of camera 2 relative to camera 1) =\n", R)
    print(f"   -> rotation angle = {ang:.1f} deg")
    print("t (unit direction from camera 1 to camera 2) =",
          np.round(t / np.linalg.norm(t), 4))
    print(f"reprojection error: img1 mean={e1.mean():.3f}px  "
          f"img2 mean={e2.mean():.3f}px")

    # save the seed pose (points are formalised in Step 5)
    out = {
        "seed_pair": [IMG_A, IMG_B],
        "focal_assumption": {
            IMG_A: fA, IMG_B: fB,
            "note": "f = 1.2*max(w,h); two views do not fix focal - refine in Step 7",
        },
        "cameras": {
            IMG_A: {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
            IMG_B: {"R": R.tolist(), "t": t.tolist()},
        },
        "shared_point_ids": pids,
        "reprojection_error_px": {"img1_mean": float(e1.mean()),
                                  "img2_mean": float(e2.mean())},
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved seed pose -> {OUT}")


if __name__ == "__main__":
    main()
