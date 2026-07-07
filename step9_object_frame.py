"""
step9_object_frame.py
Step 9 - Re-express the whole reconstruction in the POINTS' OWN coordinate frame.

The reconstruction so far floats: it was anchored to camera 01, an arbitrary
choice. This final step pins it to the OBJECT itself, so the numbers are
meaningful:

  1. ORIGIN  = the average position of all the points (their centre).
  2. AXES    = the directions the points spread out in the most. We find these
               from how the points scatter around their centre (the principal
               directions of that scatter, i.e. PCA). Strongest spread -> X,
               next -> Y, least -> Z.
  3. SCALE   = would come from one known real-world distance; we have none, so
               the size stays arbitrary (the SHAPE is correct regardless).

The same shift+turn is applied to every CAMERA too, so cameras and points end up
in one common, object-centred frame. Because it is only a rigid move (shift +
rotation), every predicted pixel is unchanged - the reconstruction is identical,
just described from a better viewpoint. (We check this: reprojection error must
be the same before and after.)

Point transform:   P_new = Rw^T (P - c)
Camera transform:  R_new = R Rw,   t_new = R c + t     (centre = -R_new^T t_new)

Inputs : data/cameras.json, data/points3d.json, data/errors.json, intrinsics
Output : data/reconstruction.json  (the final, object-framed result)

Run:
    .venv/Scripts/python.exe step9_object_frame.py
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
ERRS = os.path.join(ROOT, "data", "errors.json")
OUT = os.path.join(ROOT, "data", "reconstruction.json")


def main() -> None:
    corr = json.load(open(CORR, encoding="utf-8"))
    intr = json.load(open(INTR, encoding="utf-8"))
    cams_in = json.load(open(CAMS, encoding="utf-8"))
    cloud_in = json.load(open(CLOUD, encoding="utf-8"))
    errs = json.load(open(ERRS, encoding="utf-8"))["points"]
    marks = corr["marks"]

    pt_names = list(cloud_in["points"])
    P = np.array([cloud_in["points"][p]["XYZ"] for p in pt_names])

    # ---- build the object-centred frame from the point cloud ----
    c = P.mean(axis=0)                              # origin = centroid
    Pc = P - c
    cov = Pc.T @ Pc
    eigval, eigvec = np.linalg.eigh(cov)            # ascending
    order = np.argsort(eigval)[::-1]                # strongest spread first
    Rw = eigvec[:, order]                           # columns = X, Y, Z axes
    if np.linalg.det(Rw) < 0:                       # keep a right-handed frame
        Rw[:, 2] = -Rw[:, 2]
    spread = np.sqrt(eigval[order] / len(P))        # extent along X, Y, Z

    # ---- transform points and cameras ----
    P_new = (Rw.T @ Pc.T).T

    def K_of(im, f):
        return geo.build_K(f, intr[im]["cx"], intr[im]["cy"])

    # sanity: reprojection error must be identical after a rigid re-frame
    def reproj_all(cams, pts):
        errs_px = []
        for im, cc in cams.items():
            P_mat = K_of(im, cc["f"]) @ np.c_[cc["R"], cc["t"]]
            for i, p in enumerate(pt_names):
                if im in marks.get(p, {}):
                    errs_px.append(geo.reprojection_error(
                        P_mat, pts[i][None, :], np.array([marks[p][im]], float))[0])
        return np.mean(errs_px)

    cams_old = {im: {"R": np.array(cc["R"]), "t": np.array(cc["t"]), "f": cc["f"]}
                for im, cc in cams_in["cameras"].items()}
    cams_new = {}
    for im, cc in cams_old.items():
        R_new = cc["R"] @ Rw
        t_new = cc["R"] @ c + cc["t"]
        cams_new[im] = {"R": R_new, "t": t_new, "f": cc["f"]}

    before = reproj_all(cams_old, P)
    after = reproj_all(cams_new, P_new)
    print(f"reprojection mean: before={before:.4f}px  after={after:.4f}px  "
          f"(must match -> rigid re-frame is correct)")

    # ---- report ----
    print(f"\norigin set at cloud centroid (old frame): "
          f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    print(f"cloud extent along new X, Y, Z axes: "
          f"({spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f})  (arbitrary units)")
    print("\ncamera centres in the object frame (X=widest spread, Z=depth):")
    for im in cams_new:
        C = -cams_new[im]["R"].T @ cams_new[im]["t"]
        prov = "  [provisional]" if cams_in["cameras"][im].get("provisional") else ""
        print(f"   {im:26s} ({C[0]:7.3f}, {C[1]:7.3f}, {C[2]:7.3f}){prov}")

    # ---- save the final reconstruction ----
    out = {
        "frame": "object-centred: origin = point centroid; axes = point spread "
                 "directions (X widest, Z least); scale arbitrary (no known length)",
        "cloud_extent_xyz": spread.tolist(),
        "n_cameras": len(cams_new),
        "n_points": len(pt_names),
        "unregistered": cams_in["unregistered"],
        "cameras": {
            im: {"R": cams_new[im]["R"].tolist(),
                 "t": cams_new[im]["t"].tolist(),
                 "f": cams_new[im]["f"],
                 "center": (-cams_new[im]["R"].T @ cams_new[im]["t"]).tolist(),
                 "reproj_mean_px": cams_in["cameras"][im]["reproj_mean_px"],
                 "provisional": cams_in["cameras"][im].get("provisional", False)}
            for im in cams_new
        },
        "points": {
            p: {"XYZ": P_new[i].tolist(),
                "n_views": errs.get(p, {}).get("n_views"),
                "reproj_px": errs.get(p, {}).get("reproj_px"),
                "pos_std_rel": errs.get(p, {}).get("pos_std_rel")}
            for i, p in enumerate(pt_names)
        },
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved final reconstruction -> {OUT}")


if __name__ == "__main__":
    main()
