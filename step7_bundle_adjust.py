"""
step7_bundle_adjust.py
Step 7 - Bundle adjustment: refine ALL cameras and ALL 3D points together.

Steps 4-6 built the reconstruction piece by piece, so small errors piled up and
every focal length is still just the 1.2*max(w,h) guess. Bundle adjustment fixes
everything at once: it nudges every unknown - each camera's rotation, position
and FOCAL LENGTH, and every point's (X,Y,Z) - a little at a time, always downhill,
until the total reprojection error is as small as it can be.

The measure it minimizes is the sum, over every (point, photo) it was clicked in,
of the squared gap between the predicted pixel and the clicked pixel:

      minimize  SUM  || project(camera_c, point_p) - clicked_pixel ||^2

The engine is Levenberg-Marquardt (the same feel-the-slope-then-jump idea as
refine_pose, but now over ALL unknowns jointly). Camera 01 is held fixed as the
anchor so the scene cannot drift/spin freely; its focal is kept fixed too.

Inputs : data/cameras.json (Step 6/6b), data/points3d.json, correspondences,
         camera_intrinsics.json
Outputs: the same two files, refined in place (focals now solved, not guessed).

Run:
    .venv/Scripts/python.exe step7_bundle_adjust.py
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

ANCHOR = "01_front_clean.jpg"   # its POSE is held fixed: fixes free drift/spin
CLEAN_MAX = 10.0                # reproj below this = drop the "provisional" flag
MIN_TRI_ANGLE = 1.5             # drop points whose rays are more parallel than
                                # this (deg): their depth is unconstrained and
                                # bundle adjustment slides them off to infinity
FOCAL_PRIOR = 8.0               # gentle log-space pull keeping focals sane
                                # (negligible when data is firm, decisive on the
                                #  focal-depth degeneracy that else runs to inf)


def main() -> None:
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    cams_in = json.load(open(CAMS, encoding="utf-8"))
    cloud_in = json.load(open(CLOUD, encoding="utf-8"))
    marks = corr["marks"]

    # ---- assemble the state ----
    cam_names = list(cams_in["cameras"])
    cam = {im: {"R": np.array(c["R"]), "t": np.array(c["t"]), "f": c["f"],
                "cx": intr[im]["cx"], "cy": intr[im]["cy"]}
           for im, c in cams_in["cameras"].items()}
    pt_names = list(cloud_in["points"])
    pt_idx = {p: i for i, p in enumerate(pt_names)}
    P = np.array([cloud_in["points"][p]["XYZ"] for p in pt_names])

    # observations grouped by camera: {cam: (point_indices, uv array)}
    obs = {}
    n_obs = 0
    for im in cam_names:
        pis, uvs = [], []
        for p in pt_names:
            if im in marks.get(p, {}):
                pis.append(pt_idx[p]); uvs.append(marks[p][im])
        obs[im] = (np.array(pis), np.array(uvs, float))
        n_obs += len(pis)
    free_cams = [im for im in cam_names if im != ANCHOR]  # pose (r,t) optimized
    fguess = {im: cams_in["cameras"][im]["f"] for im in cam_names}  # 1.2*max guess
    R_anchor = cam[ANCHOR]["R"]                            # fixed (identity)
    t_anchor = cam[ANCHOR]["t"]                            # fixed (zero)
    f_anchor = cam[ANCHOR]["f"]                            # fixed -> pins scale
    # a camera's focal is refined ONLY if it is well constrained; the provisional
    # oblique views' focals are unobservable (focal-depth degeneracy) so we hold
    # them at the guess and refine only their pose.
    free_focal = [im for im in free_cams
                  if not cams_in["cameras"][im].get("provisional")]
    ff_set = set(free_focal)
    print(f"cameras: {len(cam_names)} ({ANCHOR} fully anchored to pin scale)")
    print(f"focals refined: {len(free_focal)} well-constrained; "
          f"{len(free_cams) - len(free_focal)} held (provisional oblique)")
    print(f"points: {len(pt_names)}   observations: {n_obs}")
    n_params = len(free_cams) * 6 + len(free_focal) + len(pt_names) * 3
    print(f"unknowns being solved: {n_params}\n")

    # ---- pack / unpack the parameter vector ----
    # layout: [ per free cam: rotvec(3), t(3), (f if focal free) ] [ points xyz ]
    def pack(cam, P):
        v = []
        for im in free_cams:
            c = cam[im]
            v += list(geo.rotation_log(c["R"])) + list(c["t"])
            if im in ff_set:
                v += [c["f"]]
        v += list(P.ravel())
        return np.array(v)

    def unpack(v):
        cam2 = {ANCHOR: {"R": R_anchor, "t": t_anchor, "f": f_anchor,
                         "cx": cam[ANCHOR]["cx"], "cy": cam[ANCHOR]["cy"]}}
        k = 0
        for im in free_cams:
            r = v[k:k + 3]; t = v[k + 3:k + 6]; k += 6
            if im in ff_set:
                f = v[k]; k += 1
            else:
                f = fguess[im]                            # held at the guess
            cam2[im] = {"R": geo._rodrigues(r), "t": t, "f": f,
                        "cx": cam[im]["cx"], "cy": cam[im]["cy"]}
        P2 = v[k:].reshape(-1, 3)
        return cam2, P2

    def residuals(v):
        cam2, P2 = unpack(v)
        r = []
        for im in cam_names:                              # pixel residuals
            pis, uvs = obs[im]
            c = cam2[im]
            Xc = (c["R"] @ P2[pis].T).T + c["t"]
            u = c["f"] * Xc[:, 0] / Xc[:, 2] + c["cx"]
            vv = c["f"] * Xc[:, 1] / Xc[:, 2] + c["cy"]
            r += list(u - uvs[:, 0]); r += list(vv - uvs[:, 1])
        for im in free_focal:                             # focal prior (kept gentle)
            r.append(FOCAL_PRIOR * np.log(max(cam2[im]["f"], 1e-6) / fguess[im]))
        return np.array(r)

    def rms(v):
        r = residuals(v)[:2 * n_obs].reshape(-1, 2)       # pixels only
        return np.sqrt((r ** 2).sum(1)).mean()

    # ---- Levenberg-Marquardt over the whole vector ----
    v = pack(cam, P)
    print(f"reprojection mean BEFORE: {rms(v):.3f}px")
    r0 = residuals(v); cost = r0 @ r0
    lam = 1e-3
    eps = 1e-6
    for it in range(60):
        # numeric Jacobian (columns = one perturbed parameter each)
        J = np.zeros((len(r0), len(v)))
        for j in range(len(v)):
            vv = v.copy(); vv[j] += eps
            J[:, j] = (residuals(vv) - r0) / eps
        H = J.T @ J
        g = J.T @ r0
        diag = np.diag(np.diag(H))
        stepped = False
        for _try in range(12):
            try:
                delta = np.linalg.solve(H + lam * diag, -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            vn = v + delta
            rn = residuals(vn); cn = rn @ rn
            if cn < cost:
                v, r0, cost = vn, rn, cn
                lam = max(lam * 0.5, 1e-9); stepped = True
                break
            lam *= 10
        if not stepped or np.linalg.norm(delta) < 1e-8:
            break
    print(f"reprojection mean AFTER : {rms(v):.3f}px   ({it + 1} iterations)\n")

    # ---- unpack, then drop points with too little parallax ----
    cam2, P2 = unpack(v)
    centers = {im: -cam2[im]["R"].T @ cam2[im]["t"] for im in cam_names}

    def tri_angle(p, X):
        rays = []
        for im in cam_names:
            if im in marks.get(p, {}):
                d = X - centers[im]
                rays.append(d / np.linalg.norm(d))
        best = 0.0
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                best = max(best, np.degrees(np.arccos(
                    np.clip(rays[i] @ rays[j], -1, 1))))
        return best

    kept = [p for p in pt_names if tri_angle(p, P2[pt_idx[p]]) >= MIN_TRI_ANGLE]
    dropped = [p for p in pt_names if p not in kept]
    if dropped:
        print(f"dropped {len(dropped)} low-parallax points (depth unconstrained): "
              f"{dropped}\n")

    # ---- report, save ----
    print(f"{'camera':26s} {'focal: was -> now':>22s}   {'reproj now':>10s}")
    cams_out = {"n_registered": len(cam_names),
                "n_total_photos": cams_in["n_total_photos"],
                "unregistered": cams_in["unregistered"], "cameras": {}}
    for im in cam_names:
        kp = [pt_idx[p] for p in kept if im in marks.get(p, {})]
        kuv = np.array([marks[p][im] for p in kept if im in marks.get(p, {})], float)
        c = cam2[im]
        Pm = geo.build_K(c["f"], c["cx"], c["cy"]) @ np.c_[c["R"], c["t"]]
        e = geo.reprojection_error(Pm, P2[kp], kuv).mean()
        was = cams_in["cameras"][im]["f"]
        prov = bool(e > CLEAN_MAX)
        flag = "  [still provisional]" if prov else ""
        print(f"{im:26s} {was:8.1f} -> {c['f']:8.1f}   {e:8.2f}px{flag}")
        cams_out["cameras"][im] = {
            "R": c["R"].tolist(), "t": c["t"].tolist(), "f": float(c["f"]),
            "n_pts_used": cams_in["cameras"][im].get("n_pts_used"),
            "reproj_mean_px": float(e), "provisional": prov}
    json.dump(cams_out, open(CAMS, "w", encoding="utf-8"), indent=2)

    pts_out = {}
    for p in kept:
        Xp = P2[pt_idx[p]]
        errs = []
        for im in cam_names:
            if im in marks.get(p, {}):
                c = cam2[im]
                Pm = geo.build_K(c["f"], c["cx"], c["cy"]) @ np.c_[c["R"], c["t"]]
                errs.append(float(geo.reprojection_error(
                    Pm, Xp[None, :], np.array([marks[p][im]], float))[0]))
        pts_out[p] = {"XYZ": Xp.tolist(), "n_views": len(errs),
                      "reproj_mean_px": float(np.mean(errs))}
    cloud_out = {"note": "cloud after Step 7 bundle adjustment; focals refined; "
                         "low-parallax points dropped; arbitrary scale (anchor cam 01)",
                 "n_points": len(kept), "points": pts_out}
    json.dump(cloud_out, open(CLOUD, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved refined -> {CAMS}\nSaved refined -> {CLOUD}")


if __name__ == "__main__":
    main()
