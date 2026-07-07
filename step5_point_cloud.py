"""
step5_point_cloud.py
Step 5 - Triangulate the first cloud of 3D points from the seed pair.

Each point marked in BOTH seed photos gives a line of sight out of each
camera through the clicked pixel. The 3D point is where those two lines meet
(or, since clicks are slightly off, the point closest to both lines). This is
triangulation - the same idea as fixing a location from two compass bearings.

Inputs (produced by earlier steps):
    data/correspondences.json  - the marked pixels (Step 1)
    data/camera_intrinsics.json- image centre + size per photo (Step 3)
    data/seed_pose.json        - R, t and assumed focals for the seed pair (Step 4)

Output:
    data/points3d.json         - the first 3D cloud (id -> X,Y,Z + reproj error)

The cloud is correct in SHAPE but in an ARBITRARY scale: two views cannot fix
real-world size. It is anchored to a meaningful frame later, in Step 9.

Run:
    .venv/Scripts/python.exe step5_point_cloud.py
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
POSE = os.path.join(ROOT, "data", "seed_pose.json")
OUT  = os.path.join(ROOT, "data", "points3d.json")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    pose = json.load(open(POSE, encoding="utf-8"))
    marks = corr["marks"]

    IMG_A, IMG_B = pose["seed_pair"]
    print(f"Seed pair : {IMG_A}  +  {IMG_B}")

    # shared points (same features seen in both seed photos)
    shared = [(pid, marks[pid][IMG_A], marks[pid][IMG_B])
              for pid in corr["point_ids"]
              if IMG_A in marks.get(pid, {}) and IMG_B in marks.get(pid, {})]
    pids = [s[0] for s in shared]
    x1 = np.array([s[1] for s in shared], float)
    x2 = np.array([s[2] for s in shared], float)
    print(f"Shared marked points: {len(shared)}\n")

    # rebuild the two cameras exactly as Step 4 left them
    fA = pose["focal_assumption"][IMG_A]
    fB = pose["focal_assumption"][IMG_B]
    K1 = geo.build_K(fA, intr[IMG_A]["cx"], intr[IMG_A]["cy"])
    K2 = geo.build_K(fB, intr[IMG_B]["cx"], intr[IMG_B]["cy"])
    R = np.array(pose["cameras"][IMG_B]["R"])
    t = np.array(pose["cameras"][IMG_B]["t"])

    # projection matrices  P = K [R | t]   (camera 1 is the anchor: [I | 0])
    P1 = K1 @ np.eye(3, 4)
    P2 = K2 @ np.c_[R, t]

    # triangulate every shared point
    X = geo.triangulate(P1, P2, x1, x2)

    # quality: reprojection error, and the cheirality (in-front) check
    e1 = geo.reprojection_error(P1, X, x1)
    e2 = geo.reprojection_error(P2, X, x2)
    depth1 = X[:, 2]
    depth2 = (R @ X.T + t[:, None]).T[:, 2]
    in_front = int(((depth1 > 0) & (depth2 > 0)).sum())

    # report
    print(f"{'id':>4} {'X':>8} {'Y':>8} {'Z':>8}   {'err1':>6} {'err2':>6}")
    for p, xyz, a, b in zip(pids, X, e1, e2):
        print(f"{p:>4} {xyz[0]:8.3f} {xyz[1]:8.3f} {xyz[2]:8.3f}   {a:6.2f} {b:6.2f}")
    print(f"\nreprojection error : img1 mean={e1.mean():.2f}px max={e1.max():.2f}px  "
          f"| img2 mean={e2.mean():.2f}px max={e2.max():.2f}px")
    print(f"points in front of BOTH cameras : {in_front}/{len(shared)}")
    print(f"depth (Z) range : {X[:, 2].min():.2f} to {X[:, 2].max():.2f}")

    # save the cloud
    out = {
        "seed_pair": [IMG_A, IMG_B],
        "coordinate_frame": f"camera 1 ({IMG_A}) at origin, looking down +Z",
        "scale_note": "arbitrary scale - two views do not fix real size; "
                      "anchor in Step 9",
        "points": {
            p: {"XYZ": xyz.tolist(),
                "reproj_px": {"img1": float(a), "img2": float(b)}}
            for p, xyz, a, b in zip(pids, X, e1, e2)
        },
        "summary": {
            "n_points": len(shared),
            "reproj_mean_px": {"img1": float(e1.mean()), "img2": float(e2.mean())},
            "reproj_max_px": {"img1": float(e1.max()), "img2": float(e2.max())},
            "in_front": f"{in_front}/{len(shared)}",
            "depth_range": [float(X[:, 2].min()), float(X[:, 2].max())],
        },
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved 3D cloud -> {OUT}")


if __name__ == "__main__":
    main()
