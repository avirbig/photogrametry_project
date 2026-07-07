"""
step6_register_cameras.py
Step 6 - Add the remaining photos one at a time (resection + triangulation).

Starting from the two seed cameras and the 24-point seed cloud (Steps 4-5),
grow the reconstruction photo by photo:

  6.1 CHOOSE  the unplaced photo that sees the MOST points already in the cloud
  6.2 GATHER  its (pixel <-> known 3D point) pairs
  6.3 RESECT  solve that camera's pose R, t from those pairs (geometry.resect)
  6.4 CHECK   reprojection error + points in front; register the camera
  6.5 GROW    triangulate the new points it shares with earlier cameras
  repeat until no unplaced photo sees enough known points.

This is the mirror of Step 5: there the cameras were known and we solved for
points; here the points are known and we solve for the camera.

Inputs : data/correspondences.json, data/camera_intrinsics.json,
         data/seed_pose.json (Step 4), data/points3d.json (Step 5)
Outputs: data/cameras.json   - every registered camera's R, t, focal + stats
         data/points3d.json  - the cloud, grown to cover the new cameras

Run:
    .venv/Scripts/python.exe step6_register_cameras.py
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
CLOUD = os.path.join(ROOT, "data", "points3d.json")
OUT_CAMS = os.path.join(ROOT, "data", "cameras.json")

MIN_PTS = 6      # linear resection needs >= 6 known points for a stable solve
MAX_REPROJ = 10.0  # reject a resection whose mean pixel error exceeds this
MIN_FRONT = 0.9    # and that does not put most points in front of the camera


def assumed_focal(intr, img) -> float:
    """Same normal-lens guess used for the seed pair (refined later in Step 7)."""
    return 1.2 * max(intr[img]["width"], intr[img]["height"])


def build_P(K, R, t):
    return K @ np.c_[R, t]


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    pose = json.load(open(POSE, encoding="utf-8"))
    cloud_in = json.load(open(CLOUD, encoding="utf-8"))
    marks = corr["marks"]
    all_imgs = sorted({im for p in marks.values() for im in p})

    # ---- initial state from Steps 4-5 ----
    cams = {}                       # img -> dict(R, t, f, K, n_pts, reproj)
    for img in pose["seed_pair"]:
        f = pose["focal_assumption"][img]
        K = geo.build_K(f, intr[img]["cx"], intr[img]["cy"])
        R = np.array(pose["cameras"][img]["R"])
        t = np.array(pose["cameras"][img]["t"])
        cams[img] = {"R": R, "t": t, "f": f, "K": K}
    cloud = {p: np.array(d["XYZ"]) for p, d in cloud_in["points"].items()}
    print(f"start: {len(cams)} seed cameras, {len(cloud)} points")
    print(f"seed pair: {pose['seed_pair']}\n")

    def center(c):
        return -c["R"].T @ c["t"]

    # ---- the incremental loop ----
    failed = set()   # resections rejected this round; retried after cloud grows
    while True:
        remaining = [im for im in all_imgs if im not in cams and im not in failed]
        # 6.1 pick the unplaced photo seeing the most known points
        best, best_pts = None, []
        for im in remaining:
            seen = [p for p in cloud if im in marks.get(p, {})]
            if len(seen) > len(best_pts):
                best, best_pts = im, seen
        if best is None or len(best_pts) < MIN_PTS:
            break

        # 6.2 gather 2D<->3D pairs
        X = np.array([cloud[p] for p in best_pts])
        x = np.array([marks[p][best] for p in best_pts], float)
        f = assumed_focal(intr, best)
        K = geo.build_K(f, intr[best]["cx"], intr[best]["cy"])

        # 6.3 resection (+ built-in refinement)
        R, t = geo.resect_camera(X, x, K)

        # 6.4 accept only if it reprojects well AND puts points in front
        P = build_P(K, R, t)
        e = geo.reprojection_error(P, X, x)
        depth = (R @ X.T + t[:, None])[2]
        in_front = int((depth > 0).sum())
        if e.mean() > MAX_REPROJ or in_front < MIN_FRONT * len(best_pts):
            failed.add(best)
            print(f"REJECTED   {best:26s} from {len(best_pts):2d} pts  "
                  f"reproj mean={e.mean():6.1f}px  in-front {in_front}/{len(best_pts)}"
                  f"  (retry after cloud grows)")
            continue
        cams[best] = {"R": R, "t": t, "f": f, "K": K,
                      "n_pts": len(best_pts), "reproj": float(e.mean())}
        failed.clear()   # a bigger cloud may now rescue earlier rejects
        print(f"registered {best:26s} from {len(best_pts):2d} pts  "
              f"reproj mean={e.mean():5.2f}px max={e.max():5.2f}px  "
              f"in-front {in_front}/{len(best_pts)}")

        # 6.5 grow: triangulate points this camera shares with earlier ones
        added = 0
        for p in corr["point_ids"]:
            if p in cloud or best not in marks.get(p, {}):
                continue
            viewers = [im for im in cams if im in marks.get(p, {})]
            if len(viewers) < 2:
                continue
            # widest-baseline pair for a well-conditioned crossing
            pair, wide = None, -1.0
            for i in range(len(viewers)):
                for j in range(i + 1, len(viewers)):
                    b = np.linalg.norm(center(cams[viewers[i]]) - center(cams[viewers[j]]))
                    if b > wide:
                        wide, pair = b, (viewers[i], viewers[j])
            a, c = pair
            Pa, Pc = build_P(cams[a]["K"], cams[a]["R"], cams[a]["t"]), \
                     build_P(cams[c]["K"], cams[c]["R"], cams[c]["t"])
            xa = np.array([marks[p][a]], float)
            xc = np.array([marks[p][c]], float)
            Xp = geo.triangulate(Pa, Pc, xa, xc)[0]
            cloud[p] = Xp
            added += 1
        if added:
            print(f"    -> triangulated {added} new points (cloud now {len(cloud)})")

    # ---- report + save ----
    unreg = [im for im in all_imgs if im not in cams]
    print(f"\nregistered {len(cams)}/{len(all_imgs)} cameras, "
          f"cloud grew to {len(cloud)} points")
    if unreg:
        print(f"NOT registered (saw < {MIN_PTS} known points): {unreg}")

    cams_out = {
        "n_registered": len(cams),
        "n_total_photos": len(all_imgs),
        "unregistered": unreg,
        "cameras": {
            im: {"R": c["R"].tolist(), "t": c["t"].tolist(), "f": c["f"],
                 "n_pts_used": c.get("n_pts"), "reproj_mean_px": c.get("reproj")}
            for im, c in cams.items()
        },
    }
    json.dump(cams_out, open(OUT_CAMS, "w", encoding="utf-8"), indent=2)

    # recompute a reprojection error per point across all cameras that see it
    pts_out = {}
    for p, Xp in cloud.items():
        errs = []
        for im, c in cams.items():
            if im in marks.get(p, {}):
                P = build_P(c["K"], c["R"], c["t"])
                errs.append(float(geo.reprojection_error(
                    P, Xp[None, :], np.array([marks[p][im]], float))[0]))
        pts_out[p] = {"XYZ": Xp.tolist(),
                      "n_views": len(errs),
                      "reproj_mean_px": float(np.mean(errs)) if errs else None}
    cloud_out = {
        "note": "cloud grown by Step 6 (seed pair + resectioned cameras); "
                "arbitrary scale, anchored in Step 9",
        "n_points": len(pts_out),
        "points": pts_out,
    }
    json.dump(cloud_out, open(CLOUD, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved cameras -> {OUT_CAMS}")
    print(f"Saved grown cloud -> {CLOUD}")


if __name__ == "__main__":
    main()
