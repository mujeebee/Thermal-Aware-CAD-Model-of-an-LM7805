#!/usr/bin/env python3
"""
Thermal analysis of YOUR NX model, read from STL files.

In NX:  File > Export > STL, one body at a time (select only that body), mm units:
    pcb.stl   regulator.stl   heatsink.stl

Run:    python3 solve_from_stl.py pcb.stl regulator.stl heatsink.stl
(or a single STL file that holds 3 named solids in that order)

Needs regulator_heatsink.py in the same folder (it provides the solver + hand calc).
"""
import struct
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regulator_heatsink as rh

RES = rh.RES


def read_stl(path):
    """Return a list of solids; each is an (ntri, 3, 3) array of triangle vertices (mm)."""
    data = open(path, "rb").read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):                      # binary STL
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            arr = np.frombuffer(data, dtype=dt, count=n, offset=84)
            return [arr["v"].astype(float)]
    solids, cur = [], []                                  # ASCII STL (may hold several solids)
    for line in data.decode(errors="ignore").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "solid":
            cur = []
        elif t[0] == "vertex":
            cur.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "endsolid":
            solids.append(np.array(cur).reshape(-1, 3, 3))
            cur = []
    return solids


def voxelize(tris, origin, shape):
    """Fill the voxels inside a closed triangle mesh (ray casting along z)."""
    nx, ny, nz = shape
    ox, oy, oz = origin
    jx, jy = 1.37e-4, 2.11e-4                              # tiny jitter avoids exact edge hits
    xs = ox + (np.arange(nx) + 0.5) * RES + jx
    ys = oy + (np.arange(ny) + 0.5) * RES + jy
    cols, zs = [], []
    for t in tris:
        p0, p1, p2 = t
        a, c = p1[0] - p0[0], p1[1] - p0[1]
        b, d_ = p2[0] - p0[0], p2[1] - p0[1]
        det = a * d_ - b * c
        if abs(det) < 1e-12:                               # face parallel to the ray
            continue
        i0 = max(0, int(np.ceil((t[:, 0].min() - ox - jx) / RES - 0.5)))
        i1 = min(nx - 1, int(np.floor((t[:, 0].max() - ox - jx) / RES - 0.5)))
        j0 = max(0, int(np.ceil((t[:, 1].min() - oy - jy) / RES - 0.5)))
        j1 = min(ny - 1, int(np.floor((t[:, 1].max() - oy - jy) / RES - 0.5)))
        if i1 < i0 or j1 < j0:
            continue
        gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
        dx, dy = gx - p0[0], gy - p0[1]
        l1 = (dx * d_ - b * dy) / det
        l2 = (a * dy - dx * c) / det
        l0 = 1 - l1 - l2
        m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not m.any():
            continue
        z = l0 * p0[2] + l1 * p1[2] + l2 * p2[2]
        ii, jj = np.nonzero(m)
        cols.append((i0 + ii) * ny + (j0 + jj))
        zs.append(z[m])
    filled = np.zeros(shape, bool)
    if not cols:
        return filled, 0
    cols, zs = np.concatenate(cols), np.concatenate(zs)
    order = np.lexsort((zs, cols))
    cols, zs = cols[order], zs[order]
    uniq, start, cnt = np.unique(cols, return_index=True, return_counts=True)
    bad = 0
    for u, s0, c in zip(uniq, start, cnt):
        if c % 2:
            bad += 1                                       # open mesh in this column
            continue
        i, j = divmod(int(u), ny)
        for k in range(0, c, 2):
            k0 = int(np.ceil((zs[s0 + k] - oz) / RES - 0.5))
            k1 = int(np.floor((zs[s0 + k + 1] - oz) / RES - 0.5))
            if k1 >= k0:
                filled[i, j, max(k0, 0):k1 + 1] = True
    return filled, bad


def main(paths):
    if len(paths) == 3:
        solids = [read_stl(p)[0] for p in paths]
    elif len(paths) == 1:
        solids = read_stl(paths[0])
    else:
        sys.exit("Give 3 STL files (pcb regulator heatsink) or 1 file with 3 solids.")
    if len(solids) != 3:
        sys.exit(f"Expected 3 solids, found {len(solids)}.")
    names = ["PCB", "Regulator", "Heat sink"]
    allv = np.concatenate([s.reshape(-1, 3) for s in solids])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    print("Model bounding box (mm): x %.1f..%.1f  y %.1f..%.1f  z %.1f..%.1f"
          % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    origin = np.floor(lo / RES) * RES - 1.0
    shape = tuple(int(np.ceil((hi[i] - origin[i]) / RES)) + 2 for i in range(3))

    mat = np.zeros(shape, np.int8)
    codes = [rh.FR4, rh.PKGM, rh.AL]
    expected = {}
    for nm, code, tris in zip(names, codes, solids):
        f, bad = voxelize(tris, origin, shape)
        mat[f & (mat == 0)] = code if code != rh.PKGM else 0
        if code == rh.PKGM:
            mat[f] = code
        vol = f.sum() * RES ** 3
        print("  %-10s volume %8.0f mm3 (voxels)%s" % (nm, vol, "  <-- open mesh in %d columns" % bad if bad else ""))
    print("  (expected from my dimensions: PCB 3840, regulator 405, heat sink 15300 mm3)")

    xc = origin[0] + (np.arange(shape[0]) + 0.5) * RES
    yc = origin[1] + (np.arange(shape[1]) + 0.5) * RES
    zc = origin[2] + (np.arange(shape[2]) + 0.5) * RES
    pk = np.argwhere(mat == rh.PKGM)
    if len(pk) == 0:
        sys.exit("Regulator not found in the model.")
    x_c = xc[pk[:, 0]].mean(); z_c = zc[pk[:, 2]].mean(); y_top = yc[pk[:, 1]].max() + RES / 2
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    die = ((np.abs(X - x_c) <= 1.5) & (Y >= y_top - 2.75) & (Y < y_top - 1.75)
           & (np.abs(Z - z_c) <= 1.5) & (mat == rh.PKGM))

    print("\nSolving 3D heat conduction + convection on your NX geometry ...")
    r = rh.simulate(True, voxels=(mat, die, (xc, yc, zc)))
    hc = rh.hand_calc()
    err = 100 * (r["tj"] - hc["tj_sink"]) / (hc["tj_sink"] - rh.T_AMB)
    print("  %d cells, energy balance %.3f W out vs %.3f W in" % (r["n"], r["q_out"], rh.P_W))
    print("\n[RESULT]  Junction temperature from your NX model: %.1f C" % r["tj"])
    print("          Hand calculation: %.1f C   (difference in rise %+.1f %%)" % (hc["tj_sink"], err))
    print("          Rjc %.2f  Rcs %.2f  Rsa %.2f C/W   (hand: %.2f  %.2f  %.2f)"
          % (r["r_jc"], r["r_cs"], r["r_sa"], hc["r_jc"], hc["r_cs"], hc["r_sa"]))
    print("          Limit %.0f C -> %s" % (rh.TJ_LIMIT, "PASS" if r["tj"] <= rh.TJ_LIMIT else "FAIL"))

    k = int(np.argmin(np.abs(zc - z_c)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(r["T"][:, :, k].T, origin="lower", cmap="inferno",
                   extent=(xc[0] - RES / 2, xc[-1] + RES / 2, yc[0] - RES / 2, yc[-1] + RES / 2))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("NX model, temperature slice through regulator\npeak junction %.0f C" % r["tj"])
    fig.colorbar(im, label="Temperature (C)", shrink=0.8)
    fig.tight_layout(); fig.savefig("temperature_from_nx.png", dpi=150)
    print("\nSaved temperature_from_nx.png")


if __name__ == "__main__":
    main(sys.argv[1:])
