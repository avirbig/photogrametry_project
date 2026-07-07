"""
step8_report_errors.py
Step 8 - Read off the error values from the finished reconstruction.

Two kinds of error, both falling out of the Step 7 bundle adjustment:

  1. REPROJECTION ERROR (pixels), per point: push the 3D point back through
     every camera that saw it and average the gap to the click. Small = the
     point is consistent across photos; large = a careless click or mismatch.

  2. 3D UNCERTAINTY (scene units), per point: how far the point could move
     before the fit noticeably worsens. It comes from the STEEPNESS of the error
     bowl at the solution (the same slope/steepness idea as the refinement): a
     point pinned by many well-spread cameras sits in a steep, narrow bowl (small
     wobble); a point seen by few cameras at a shallow angle sits in a flat bowl
     (large wobble). Formally the point's position covariance is

         Cov = sigma0^2 * (J^T J)^-1   restricted to that point's 3 coordinates
         sigma0^2 = (sum of squared pixel residuals) / (2M - U)

     where 2M is the number of measured pixel-numbers and U the number of
     unknowns (the fit's "degrees of freedom"). The square roots of that 3x3
     block's eigenvalues are the point's wobble along its three least-certain
     directions.

Scale note: the cloud has no metric scale yet (Step 9), so 3D wobble is given in
the arbitrary scene units AND as a fraction of the cloud's overall size.

Inputs : data/cameras.json, data/points3d.json, correspondences, intrinsics
Output : data/errors.json  (per-point reprojection error + 3D uncertainty)

Run:
    .venv/Scripts/python.exe step8_report_errors.py
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
OUT = os.path.join(ROOT, "data", "errors.json")

ANCHOR = "01_front_clean.jpg"


def main() -> None:
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    cams_in = json.load(open(CAMS, encoding="utf-8"))
    cloud_in = json.load(open(CLOUD, encoding="utf-8"))
    marks = corr["marks"]

    cam_names = list(cams_in["cameras"])
    cam = {im: {"R": np.array(c["R"]), "t": np.array(c["t"]), "f": c["f"],
                "cx": intr[im]["cx"], "cy": intr[im]["cy"]}
           for im, c in cams_in["cameras"].items()}
    pt_names = list(cloud_in["points"])
    pt_idx = {p: i for i, p in enumerate(pt_names)}
    P = np.array([cloud_in["points"][p]["XYZ"] for p in pt_names])

    obs = {}
    n_obs = 0
    for im in cam_names:
        pis, uvs = [], []
        for p in pt_names:
            if im in marks.get(p, {}):
                pis.append(pt_idx[p]); uvs.append(marks[p][im])
        obs[im] = (np.array(pis), np.array(uvs, float))
        n_obs += len(pis)

    # ---- same free-parameter layout as Step 7 (so J matches the solved fit) ----
    free_cams = [im for im in cam_names if im != ANCHOR]
    free_focal = set(im for im in free_cams
                     if not cams_in["cameras"][im].get("provisional"))
    # column offsets
    cam_cols = {}
    k = 0
    for im in free_cams:
        cam_cols[im] = k
        k += 6 + (1 if im in free_focal else 0)
    pt_col0 = k
    n_params = pt_col0 + 3 * len(pt_names)

    def unpack(v):
        cam2 = {ANCHOR: cam[ANCHOR]}
        for im in free_cams:
            c0 = cam_cols[im]
            r = v[c0:c0 + 3]; t = v[c0 + 3:c0 + 6]
            f = v[c0 + 6] if im in free_focal else cam[im]["f"]
            cam2[im] = {"R": geo._rodrigues(r), "t": t, "f": f,
                        "cx": cam[im]["cx"], "cy": cam[im]["cy"]}
        P2 = v[pt_col0:].reshape(-1, 3)
        return cam2, P2

    def pixel_residuals(v):
        cam2, P2 = unpack(v)
        r = []
        for im in cam_names:
            pis, uvs = obs[im]
            c = cam2[im]
            Xc = (c["R"] @ P2[pis].T).T + c["t"]
            u = c["f"] * Xc[:, 0] / Xc[:, 2] + c["cx"]
            vv = c["f"] * Xc[:, 1] / Xc[:, 2] + c["cy"]
            r += list(u - uvs[:, 0]); r += list(vv - uvs[:, 1])
        return np.array(r)

    # build the solution vector v0
    v0 = np.zeros(n_params)
    for im in free_cams:
        c0 = cam_cols[im]
        v0[c0:c0 + 3] = geo.rotation_log(cam[im]["R"])
        v0[c0 + 3:c0 + 6] = cam[im]["t"]
        if im in free_focal:
            v0[c0 + 6] = cam[im]["f"]
    v0[pt_col0:] = P.ravel()

    # ---- covariance from the steepness (J^T J) at the solution ----
    r0 = pixel_residuals(v0)
    ssr = r0 @ r0
    dof = 2 * n_obs - n_params
    sigma0 = np.sqrt(ssr / dof)                    # typical leftover gap, px
    J = np.zeros((len(r0), n_params))
    eps = 1e-6
    for j in range(n_params):
        vv = v0.copy(); vv[j] += eps
        J[:, j] = (pixel_residuals(vv) - r0) / eps
    JtJ = J.T @ J
    cov = sigma0 ** 2 * np.linalg.pinv(JtJ)         # pinv: robust if near-singular

    # ---- cloud size, for relative uncertainty ----
    spread = np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())

    # ---- per-point numbers ----
    rows = []
    for p in pt_names:
        i = pt_idx[p]
        block = cov[pt_col0 + 3 * i: pt_col0 + 3 * i + 3,
                    pt_col0 + 3 * i: pt_col0 + 3 * i + 3]
        eig = np.clip(np.linalg.eigvalsh(block), 0, None)
        std = np.sqrt(eig)                          # wobble along 3 axes
        rows.append({
            "id": p,
            "n_views": cloud_in["points"][p]["n_views"],
            "reproj_px": cloud_in["points"][p]["reproj_mean_px"],
            "pos_std_units": float(std.max()),      # worst-direction wobble
            "pos_std_rel": float(std.max() / spread),
        })

    # ---- report ----
    reproj = np.array([r["reproj_px"] for r in rows])
    wob = np.array([r["pos_std_rel"] for r in rows])
    print(f"points: {len(rows)}   sigma0 (typical pixel gap): {sigma0:.2f}px")
    print(f"reprojection error: mean={reproj.mean():.2f}px  median={np.median(reproj):.2f}px  max={reproj.max():.2f}px")
    print(f"3D wobble (fraction of cloud size): median={np.median(wob)*100:.1f}%  max={wob.max()*100:.1f}%\n")
    worst_r = sorted(rows, key=lambda r: -r["reproj_px"])[:5]
    worst_w = sorted(rows, key=lambda r: -r["pos_std_rel"])[:5]
    print("least reliable by REPROJECTION (px):")
    for r in worst_r:
        print(f"   {r['id']:5s} {r['reproj_px']:6.2f}px  seen in {r['n_views']} views")
    print("least reliable by 3D WOBBLE (fewest/weakest views):")
    for r in worst_w:
        print(f"   {r['id']:5s} wobble {r['pos_std_rel']*100:5.1f}% of cloud  "
              f"seen in {r['n_views']} views  ({r['reproj_px']:.1f}px)")

    out = {
        "sigma0_px": float(sigma0),
        "degrees_of_freedom": int(dof),
        "reproj_px": {"mean": float(reproj.mean()),
                      "median": float(np.median(reproj)),
                      "max": float(reproj.max())},
        "note": "pos_std is 3D wobble in arbitrary scene units and as a fraction "
                "of cloud size; metric scale is set in Step 9",
        "points": {r["id"]: {k: r[k] for k in
                             ("n_views", "reproj_px", "pos_std_units", "pos_std_rel")}
                   for r in rows},
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved error report -> {OUT}")


if __name__ == "__main__":
    main()
