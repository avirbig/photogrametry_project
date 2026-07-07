"""
step6b_register_oblique.py
Step 6b - Rescue the oblique views that plain resection could not place.

Step 6 registered the 9 frontal-ish cameras but rejected the 5 diagonal/side
views (07-11). The reason: the cloud built so far is near-planar (the facade),
and resection cannot recover a camera's ROTATION from near-planar 3D points.

The fix does NOT use the flat cloud. Instead, for each oblique view we pair it
with an already-registered NEIGHBOUR and recover their relative pose from the
two images' 2D matches via the essential matrix (Step 4's method) - which needs
only PARALLAX between the two photos, not a 3D cloud. Then:

  1. Pick the neighbour with the MOST parallax (highest homography residual -
     a single flat plane cannot explain the matches -> real depth is present).
     NOTE: parallax, not "most shared points" - picking most-shared for the
     original seed pair is exactly what made the cloud flat in the first place.
  2. Essential matrix -> the relative rotation R_rel (recovered robustly).
     Compose into the world frame:  R_obl = R_rel R_neighbour.
  3. With R_obl fixed, solve the translation t linearly from the cloud points
     the oblique view sees (well-conditioned even on planar points), then
     LM-refine (R, t) together.
  4. Accept on reprojection + cheirality; triangulate its OFF-plane points to
     grow the cloud. Those new non-planar points then let the remaining views
     (incl. any straggler) register - by essential matrix or, once the cloud is
     no longer flat, by plain resection again.

Inputs : data/correspondences.json, data/camera_intrinsics.json,
         data/cameras.json (Step 6), data/points3d.json (Step 6 cloud)
Outputs: the same two files, updated in place with the rescued cameras/points.

Run:
    .venv/Scripts/python.exe step6b_register_oblique.py
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
CAMS = os.path.join(ROOT, "data", "cameras.json")
CLOUD = os.path.join(ROOT, "data", "points3d.json")

MIN_SHARED = 8       # essential matrix needs about 8 matches
PARALLAX_MIN = 4.0   # median homography residual (px) below this = too planar
MAX_REPROJ = 10.0    # a "clean" registration
PROVISIONAL_MAX = 45.0  # geometrically valid but focal-limited; refine in Step 7
MIN_FRONT = 0.9


def assumed_focal(intr, img):
    return 1.2 * max(intr[img]["width"], intr[img]["height"])


def homography_resid(x1, x2):
    """Median pixel error of the best single-plane warp x2 ~ H x1 (parallax)."""
    n1, T1 = geo._normalize(x1)
    n2, T2 = geo._normalize(x2)
    A = []
    for (X, Y, _), (u, v, _) in zip(n1, n2):
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
    p = (H @ np.c_[x1, np.ones(len(x1))].T).T
    p = p[:, :2] / p[:, 2:]
    return float(np.median(np.sqrt(((p - x2) ** 2).sum(1))))


def main() -> None:
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    cams_in = json.load(open(CAMS, encoding="utf-8"))
    cloud_in = json.load(open(CLOUD, encoding="utf-8"))
    marks = corr["marks"]

    cams = {}
    for im, c in cams_in["cameras"].items():
        K = geo.build_K(c["f"], intr[im]["cx"], intr[im]["cy"])
        cams[im] = {"R": np.array(c["R"]), "t": np.array(c["t"]),
                    "f": c["f"], "K": K}
    cloud = {p: np.array(d["XYZ"]) for p, d in cloud_in["points"].items()}
    unreg = list(cams_in["unregistered"])
    print(f"start: {len(cams)} registered, {len(cloud)} points, "
          f"{len(unreg)} to rescue: {unreg}\n")

    def center(c):
        return -c["R"].T @ c["t"]

    def shared(a, b):
        return [p for p in corr["point_ids"]
                if a in marks.get(p, {}) and b in marks.get(p, {})]

    def grow_from(new_img):
        added = 0
        for p in corr["point_ids"]:
            if p in cloud or new_img not in marks.get(p, {}):
                continue
            viewers = [im for im in cams if im in marks.get(p, {})]
            if len(viewers) < 2:
                continue
            pair, wide = None, -1.0
            for i in range(len(viewers)):
                for j in range(i + 1, len(viewers)):
                    d = np.linalg.norm(center(cams[viewers[i]]) - center(cams[viewers[j]]))
                    if d > wide:
                        wide, pair = d, (viewers[i], viewers[j])
            a, b = pair
            Pa = cams[a]["K"] @ np.c_[cams[a]["R"], cams[a]["t"]]
            Pb = cams[b]["K"] @ np.c_[cams[b]["R"], cams[b]["t"]]
            cloud[p] = geo.triangulate(Pa, Pb,
                                       np.array([marks[p][a]], float),
                                       np.array([marks[p][b]], float))[0]
            added += 1
        return added

    def try_register(img, R, t, K, pts):
        """
        Accept a pose if points land in front and it reprojects acceptably.
        Below MAX_REPROJ it is a clean registration; between there and
        PROVISIONAL_MAX it is geometrically valid but focal-limited, kept as
        PROVISIONAL for Step 7 to refine. Returns (mean_error, provisional?).
        """
        X = np.array([cloud[p] for p in pts])
        x = np.array([marks[p][img] for p in pts], float)
        R, t = geo.refine_pose(R, t, X, x, K)
        P = K @ np.c_[R, t]
        e = geo.reprojection_error(P, X, x)
        in_front = (R @ X.T + t[:, None])[2] > 0
        if in_front.sum() < MIN_FRONT * len(pts) or e.mean() > PROVISIONAL_MAX:
            return None
        provisional = bool(e.mean() > MAX_REPROJ)
        cams[img] = {"R": R, "t": t, "f": K[0, 0], "K": K,
                     "n_pts": len(pts), "reproj": float(e.mean()),
                     "provisional": provisional}
        return e.mean(), provisional

    # ---- passes: essential-matrix rescue, then cascade ----
    while True:
        progress = False
        for img in list(unreg):
            Ku = geo.build_K(assumed_focal(intr, img), intr[img]["cx"], intr[img]["cy"])
            cloud_pts = [p for p in cloud if img in marks.get(p, {})]

            # choose the registered neighbour with the MOST parallax
            best = None
            for r in cams:
                sh = shared(img, r)
                if len(sh) < MIN_SHARED:
                    continue
                x1 = np.array([marks[p][r] for p in sh], float)
                x2 = np.array([marks[p][img] for p in sh], float)
                hr = homography_resid(x1, x2)
                if best is None or hr > best[0]:
                    best = (hr, r, sh, x1, x2)
            if best is None:
                continue
            hr, r, sh, x1, x2 = best

            placed = None
            if hr >= PARALLAX_MIN and len(cloud_pts) >= 3:
                # essential matrix -> relative rotation -> world rotation
                F = geo.fundamental_from_points(x1, x2)
                E = geo.essential_from_fundamental(F, cams[r]["K"], Ku)
                Ra, Rb, tr = geo.decompose_essential(E)
                Rrel, *_ = geo.select_pose(Ra, Rb, tr, cams[r]["K"], Ku, x1, x2)
                R_obl = Rrel @ cams[r]["R"]
                Xc = np.array([cloud[p] for p in cloud_pts])
                xc = np.array([marks[p][img] for p in cloud_pts], float)
                t_obl = geo.translation_given_rotation(R_obl, Xc, xc, Ku)
                placed = try_register(img, R_obl, t_obl, Ku, cloud_pts)
                how = f"essential vs {r} (parallax {hr:.1f}px)"

            # fallback: plain resection on the (possibly grown) cloud
            if placed is None and len(cloud_pts) >= 6:
                Xc = np.array([cloud[p] for p in cloud_pts])
                xc = np.array([marks[p][img] for p in cloud_pts], float)
                R, t = geo.resect_camera(Xc, xc, Ku)
                placed = try_register(img, R, t, Ku, cloud_pts)
                how = "plain resection on grown cloud"

            if placed is not None:
                err, provisional = placed
                unreg.remove(img)
                added = grow_from(img)
                progress = True
                tag = "PROVISIONAL" if provisional else "clean"
                print(f"rescued  {img:26s} via {how}")
                print(f"           reproj mean={err:.1f}px [{tag}]  "
                      f"+{added} new points (cloud now {len(cloud)})")
        if not progress:
            break

    # ---- save ----
    all_imgs = sorted({im for p in marks.values() for im in p})
    unreg_final = [im for im in all_imgs if im not in cams]
    print(f"\nnow {len(cams)}/{len(all_imgs)} cameras registered, "
          f"cloud {len(cloud)} points")
    if unreg_final:
        print("still unregistered:", unreg_final)

    cams_out = {
        "n_registered": len(cams),
        "n_total_photos": len(all_imgs),
        "unregistered": unreg_final,
        "cameras": {im: {"R": c["R"].tolist(), "t": c["t"].tolist(), "f": c["f"],
                         "n_pts_used": c.get("n_pts"),
                         "reproj_mean_px": c.get("reproj"),
                         "provisional": c.get("provisional", False)}
                    for im, c in cams.items()},
    }
    json.dump(cams_out, open(CAMS, "w", encoding="utf-8"), indent=2)

    pts_out = {}
    for p, Xp in cloud.items():
        errs = []
        for im, c in cams.items():
            if im in marks.get(p, {}):
                P = c["K"] @ np.c_[c["R"], c["t"]]
                errs.append(float(geo.reprojection_error(
                    P, Xp[None, :], np.array([marks[p][im]], float))[0]))
        pts_out[p] = {"XYZ": Xp.tolist(), "n_views": len(errs),
                      "reproj_mean_px": float(np.mean(errs)) if errs else None}
    cloud_out = {"note": "cloud after Step 6b oblique rescue; arbitrary scale",
                 "n_points": len(pts_out), "points": pts_out}
    json.dump(cloud_out, open(CLOUD, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved -> {CAMS}\nSaved -> {CLOUD}")


if __name__ == "__main__":
    main()
