#!/usr/bin/env python3
"""
LM7805 on a PCB with a finned aluminium heat sink
=================================================
Terminal-only workflow:
  1. Parametric 3D model  -> writes assembly.stl (open in NX / FreeCAD / any STL viewer)
  2. Hand calculation     -> thermal-resistance chain (like a resistor network)
  3. 3D thermal simulation-> steady-state heat conduction + convection (finite volume)
  4. Comparison           -> results.csv, geometry.png, temperature.png

Run:   python3 regulator_heatsink.py
Needs: numpy, scipy, matplotlib   (nothing else)

All lengths in mm unless a name ends in _m (metres).
"""
import csv
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =====================================================================
# 1. INPUTS  (change these and re-run)
# =====================================================================
# --- electrical operating point of the LM7805
VIN, VOUT, IOUT = 12.0, 5.0, 0.5          # V, V, A
T_AMB = 40.0                              # ambient, deg C
TJ_LIMIT = 100.0                          # design limit (datasheet max is 125)

# --- thermal data (take from YOUR datasheet; typical values here)
RJC_DATASHEET = 5.0                       # junction->case, deg C/W
RJA_DATASHEET = 65.0                      # junction->ambient, no sink, deg C/W
H_CONV = 10.0                             # natural convection + radiation, W/m^2K
K_AL = 200.0                              # W/mK  (6063 aluminium)
K_PKG = 15.0                              # W/mK  lumped package, CALIBRATED so simulated Rjc ~ datasheet 5 C/W
K_FR4_XY, K_FR4_Z = 17.0, 0.3             # W/mK  (copper layers help in-plane)
PAD_T_MM, PAD_K = 0.1, 1.5                # thermal grease/pad thickness, conductivity

# --- geometry (mm)
BOARD = dict(x=(0, 60), y=(0, 40), z=(0, 1.6))
PKG_W, PKG_D, PKG_H = 10.0, 4.5, 9.0      # TO-220 body (x, y, z)
PKG_CX = 30.0
SINK_BASE_T = 3.0                         # base plate thickness
SINK_W, SINK_L = 40.0, 40.0               # base size across (x) and along (z)
N_FINS, FIN_T, FIN_H = 7, 1.5, 25.0

RES = 0.5                                 # voxel size for simulation, mm

# derived geometry --------------------------------------------------------
P_W = (VIN - VOUT) * IOUT                 # heat to be removed, W
PKG = dict(x=(PKG_CX - PKG_W / 2, PKG_CX + PKG_W / 2),
           y=(BOARD["y"][1] - PKG_D, BOARD["y"][1]),
           z=(BOARD["z"][1], BOARD["z"][1] + PKG_H))
PKG_ZC = 0.5 * (PKG["z"][0] + PKG["z"][1])
SINK_X0 = PKG_CX - SINK_W / 2
SINK_Y0 = PKG["y"][1]                     # sink base touches the package face
SINK_Z0, SINK_Z1 = PKG_ZC - SINK_L / 2, PKG_ZC + SINK_L / 2
BASE = dict(x=(SINK_X0, SINK_X0 + SINK_W), y=(SINK_Y0, SINK_Y0 + SINK_BASE_T),
            z=(SINK_Z0, SINK_Z1))
PITCH = (SINK_W - FIN_T) / (N_FINS - 1)
FINS = [dict(x=(SINK_X0 + i * PITCH, SINK_X0 + i * PITCH + FIN_T),
             y=(SINK_Y0 + SINK_BASE_T, SINK_Y0 + SINK_BASE_T + FIN_H),
             z=(SINK_Z0, SINK_Z1)) for i in range(N_FINS)]
FIN_GAP = PITCH - FIN_T
# die (heat source) sits 2.25 mm inside the package, behind the tab face
DIE = dict(x=(PKG_CX - 1.5, PKG_CX + 1.5),
           y=(PKG["y"][1] - 2.75, PKG["y"][1] - 1.75),
           z=(PKG_ZC - 1.5, PKG_ZC + 1.5))
CONTACT_AREA_M2 = (PKG_W * PKG_H) * 1e-6


# =====================================================================
# 2. CAD OUTPUT : STL of the assembly
# =====================================================================
def box_triangles(b):
    (x0, x1), (y0, y1), (z0, z1) = b["x"], b["y"], b["z"]
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = []
    for a, b_, c, d in f:
        tris.append((v[a], v[b_], v[c]))
        tris.append((v[a], v[c], v[d]))
    return tris


def write_stl(path, solids):
    with open(path, "w") as fh:
        for name, boxes in solids.items():
            fh.write(f"solid {name}\n")
            for b in boxes:
                for t in box_triangles(b):
                    n = np.cross(t[1] - t[0], t[2] - t[0])
                    n = n / (np.linalg.norm(n) or 1.0)
                    fh.write(f" facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n  outer loop\n")
                    for p in t:
                        fh.write(f"   vertex {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
                    fh.write("  endloop\n endfacet\n")
            fh.write(f"endsolid {name}\n")


def draw_geometry(path, with_sink=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    groups = [("PCB (FR4)", [BOARD], "#3f7f4f"), ("LM7805 (TO-220)", [PKG], "#222222")]
    if with_sink:
        groups.append(("Heat sink (Al)", [BASE] + FINS, "#9fb3c8"))
    for name, boxes, col in groups:
        first = True
        for b in boxes:
            tris = box_triangles(b)
            poly = Poly3DCollection([[tuple(p) for p in t] for t in tris], alpha=0.35 if col == "#9fb3c8" else 0.95,
                                    facecolor=col, edgecolor="k", linewidths=0.2,
                                    label=name if first else None)
            ax.add_collection3d(poly)
            first = False
    ax.set_xlim(0, 60); ax.set_ylim(0, 70); ax.set_zlim(-15, 27)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_box_aspect((60, 70, 42))
    ax.view_init(elev=28, azim=-35)
    ax.set_title("Regulator + PCB + finned heat sink (parametric model)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =====================================================================
# 3. HAND CALCULATION  (thermal resistances in series, like resistors)
# =====================================================================
def hand_calc():
    h = H_CONV
    L = SINK_L * 1e-3
    t = FIN_T * 1e-3
    fh = FIN_H * 1e-3
    lc = fh + t / 2
    m = np.sqrt(2 * h / (K_AL * t))
    eta = np.tanh(m * lc) / (m * lc)                      # fin efficiency
    a_fin = N_FINS * (2 * fh * L + t * L) + 0.0           # both sides + tip
    base_w = SINK_W * 1e-3
    a_base = (base_w * L) * 2 - N_FINS * t * L - CONTACT_AREA_M2 \
             + 2 * (SINK_BASE_T * 1e-3) * (base_w + L)    # two faces + 4 edges
    r_sa = 1.0 / (h * (a_base + eta * a_fin))
    r_cs = (PAD_T_MM * 1e-3 / PAD_K) / CONTACT_AREA_M2
    r_jc = RJC_DATASHEET
    tj_sink = T_AMB + P_W * (r_jc + r_cs + r_sa)
    tj_nosink_datasheet = T_AMB + P_W * RJA_DATASHEET
    r_sa_required = (TJ_LIMIT - T_AMB) / P_W - r_jc - r_cs
    return dict(P=P_W, eta=eta, a_fin=a_fin, a_base=a_base, r_jc=r_jc, r_cs=r_cs,
                r_sa=r_sa, tj_sink=tj_sink, tj_nosink=tj_nosink_datasheet,
                r_sa_required=r_sa_required)


# =====================================================================
# 4. 3D THERMAL SIMULATION  (voxel finite volume, steady state)
# =====================================================================
FR4, PKGM, AL = 1, 2, 3


def build_voxels(with_sink):
    ox, oy, oz = 0.0, 0.0, -14.0
    nx = int(round(60 / RES))
    ny = int(round(70 / RES))
    nz = int(round(41 / RES))
    xc = ox + (np.arange(nx) + 0.5) * RES
    yc = oy + (np.arange(ny) + 0.5) * RES
    zc = oz + (np.arange(nz) + 0.5) * RES
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")

    def inside(b):
        return ((X >= b["x"][0]) & (X < b["x"][1]) & (Y >= b["y"][0]) & (Y < b["y"][1])
                & (Z >= b["z"][0]) & (Z < b["z"][1]))

    mat = np.zeros((nx, ny, nz), np.int8)
    mat[inside(BOARD)] = FR4
    mat[inside(PKG)] = PKGM
    if with_sink:
        mat[inside(BASE)] = AL
        for f in FINS:
            mat[inside(f)] = AL
    die = inside(DIE) & (mat == PKGM)
    return mat, die, (xc, yc, zc)


def conductivities(mat):
    kx = np.zeros(mat.shape); ky = np.zeros(mat.shape); kz = np.zeros(mat.shape)
    for code, (a, b, c) in {FR4: (K_FR4_XY, K_FR4_XY, K_FR4_Z),
                            PKGM: (K_PKG,) * 3, AL: (K_AL,) * 3}.items():
        m = mat == code
        kx[m], ky[m], kz[m] = a, b, c
    return (kx, ky, kz)


def simulate(with_sink, voxels=None):
    t0 = time.time()
    mat, die, (xc, yc, zc) = voxels if voxels is not None else build_voxels(with_sink)
    ks = conductivities(mat)
    solid = mat > 0
    n = int(solid.sum())
    idx = -np.ones(mat.shape, np.int64)
    idx[solid] = np.arange(n)
    dx = RES * 1e-3
    A = dx * dx
    rc_pad = (PAD_T_MM * 1e-3) / PAD_K        # contact resistance per unit area (m2K/W)

    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    rhs = np.zeros(n)

    # conduction between neighbouring solid cells
    for ax in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1); sl_b[ax] = slice(1, None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)
        ma, mb = mat[sl_a], mat[sl_b]
        pair = (ma > 0) & (mb > 0)
        # PCB edge and sink are not bonded: no direct conduction between them
        pair &= ~(((ma == FR4) & (mb == AL)) | ((ma == AL) & (mb == FR4)))
        ka, kb = ks[ax][sl_a][pair], ks[ax][sl_b][pair]
        rc = np.where(((ma[pair] == PKGM) & (mb[pair] == AL)) |
                      ((ma[pair] == AL) & (mb[pair] == PKGM)), rc_pad, 0.0)
        G = A / (dx / (2 * ka) + dx / (2 * kb) + rc)
        ia, ib = idx[sl_a][pair], idx[sl_b][pair]
        np.add.at(diag, ia, G); np.add.at(diag, ib, G)
        rows += [ia, ib]; cols += [ib, ia]; vals += [-G, -G]

    # convection from every solid face that sees air (or the outside)
    conv_G = np.zeros(n)
    for ax in range(3):
        for side in (0, 1):
            shifted = np.zeros(mat.shape, bool)
            sl_dst = [slice(None)] * 3; sl_src = [slice(None)] * 3
            if side == 0:       # neighbour at lower index
                sl_dst[ax] = slice(1, None); sl_src[ax] = slice(0, -1)
                shifted[tuple(sl_dst)] = solid[tuple(sl_src)]
                shifted_air = ~shifted           # True when lower neighbour is not solid
                edge = np.zeros(mat.shape, bool)
                e = [slice(None)] * 3; e[ax] = slice(0, 1); edge[tuple(e)] = True
            else:               # neighbour at higher index
                sl_dst[ax] = slice(0, -1); sl_src[ax] = slice(1, None)
                shifted[tuple(sl_dst)] = solid[tuple(sl_src)]
                shifted_air = ~shifted
                edge = np.zeros(mat.shape, bool)
                e = [slice(None)] * 3; e[ax] = slice(-1, None); edge[tuple(e)] = True
            exposed = solid & shifted_air
            G = A / (dx / (2 * ks[ax][exposed]) + 1.0 / H_CONV)
            ii = idx[exposed]
            np.add.at(diag, ii, G)
            np.add.at(conv_G, ii, G)
            np.add.at(rhs, ii, G * T_AMB)

    # heat source: P_W spread uniformly over the die cells
    ndie = int(die.sum())
    rhs[idx[die]] += P_W / ndie

    rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    # solve  M T = rhs  (symmetric positive definite)
    d = M.diagonal()
    T_sol, info = spla.cg(M, rhs, x0=np.full(n, T_AMB), rtol=1e-10, atol=0.0,
                          maxiter=40000, M=sp.diags(1.0 / d))
    if info != 0:
        T_sol = spla.spsolve(M.tocsc(), rhs)
    T = np.full(mat.shape, np.nan)
    T[solid] = T_sol

    # ---- energy balance check: heat leaving by convection must equal P_W
    q_out = float(np.sum(conv_G * (T_sol - T_AMB)))

    res = dict(T=T, mat=mat, xc=xc, yc=yc, zc=zc, n=n, q_out=q_out,
               tj=float(np.nanmax(np.where(die, T, np.nan))),
               t_max=float(np.nanmax(T)), seconds=time.time() - t0)

    if with_sink:
        # thermal circuit extracted from the simulation
        pk = mat == PKGM
        # package cells touching the sink = package cells at last y layer
        face_y = np.where(pk.any(axis=(0, 2)))[0].max()
        t_case = float(np.nanmean(T[:, face_y, :][pk[:, face_y, :]]))
        base = (mat == AL)
        y_base0 = face_y + 1
        sel = pk[:, face_y, :]
        t_sink_c = float(np.nanmean(T[:, y_base0, :][sel & base[:, y_base0, :]]))
        res.update(t_case=t_case, t_sink_contact=t_sink_c,
                   r_jc=(res["tj"] - t_case) / P_W,
                   r_cs=(t_case - t_sink_c) / P_W,
                   r_sa=(t_sink_c - T_AMB) / P_W)
    return res


def draw_temperature(path, r_sink, r_none):
    zc_pkg = PKG_ZC
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, r, title in ((axs[0], r_sink, "WITH heat sink"), (axs[1], r_none, "NO heat sink")):
        k = int(np.argmin(np.abs(r["zc"] - zc_pkg)))
        sl = r["T"][:, :, k].T
        extent = (0, 60, 0, 70)
        im = ax.imshow(sl, origin="lower", extent=extent, cmap="inferno", aspect="equal")
        ax.set_title(f"{title}: peak junction {r['tj']:.0f} °C")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, label="Temperature (°C)", shrink=0.8)
    fig.suptitle(f"Temperature slice through the regulator (z = {zc_pkg:.1f} mm), "
                 f"{P_W:.1f} W dissipated, ambient {T_AMB:.0f} °C")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =====================================================================
# 5. MAIN
# =====================================================================
def main():
    print("=" * 62)
    print(f"LM7805: Vin {VIN} V, Vout {VOUT} V, I {IOUT} A  ->  P = {P_W:.2f} W")
    print(f"Ambient {T_AMB} C, junction limit {TJ_LIMIT} C")
    print("=" * 62)

    # CAD
    write_stl("assembly.stl", {
        "pcb": [BOARD], "lm7805": [PKG], "heatsink": [BASE] + FINS})
    draw_geometry("geometry.png")
    print(f"[CAD ] assembly.stl + geometry.png written. fin gap = {FIN_GAP:.2f} mm")

    # hand calc
    hc = hand_calc()
    print("\n[HAND CALCULATION]  (thermal resistances in series)")
    print(f"  no sink (datasheet RthJA {RJA_DATASHEET} C/W): Tj = {hc['tj_nosink']:.0f} C")
    print(f"  required heat-sink resistance for Tj<={TJ_LIMIT:.0f} C : Rsa <= {hc['r_sa_required']:.1f} C/W")
    print(f"  fin efficiency            = {hc['eta']:.3f}")
    print(f"  Rjc (datasheet)           = {hc['r_jc']:.2f} C/W")
    print(f"  Rcs (pad)                 = {hc['r_cs']:.2f} C/W")
    print(f"  Rsa (sink, fin formula)   = {hc['r_sa']:.2f} C/W")
    print(f"  Tj with sink              = {hc['tj_sink']:.1f} C")

    # simulation
    print("\n[SIMULATION] solving with-sink case ...")
    rs = simulate(True)
    print(f"  {rs['n']:,} cells, {rs['seconds']:.0f} s, energy balance: "
          f"{rs['q_out']:.3f} W out vs {P_W:.3f} W in")
    print("[SIMULATION] solving no-sink case ...")
    rn = simulate(False)
    print(f"  {rn['n']:,} cells, {rn['seconds']:.0f} s, energy balance: "
          f"{rn['q_out']:.3f} W out vs {P_W:.3f} W in")
    draw_temperature("temperature.png", rs, rn)

    print("\n[RESULTS]")
    print(f"  WITH sink  : simulated Tj = {rs['tj']:.1f} C   (hand calc {hc['tj_sink']:.1f} C)")
    err = 100 * (rs["tj"] - hc["tj_sink"]) / (hc["tj_sink"] - T_AMB)
    print(f"               difference in temperature rise = {err:+.1f} %")
    print(f"               simulated Rjc {rs['r_jc']:.2f}, Rcs {rs['r_cs']:.2f}, "
          f"Rsa {rs['r_sa']:.2f} C/W   (hand: {hc['r_jc']:.2f}, {hc['r_cs']:.2f}, {hc['r_sa']:.2f})")
    print(f"  NO sink    : simulated Tj = {rn['tj']:.0f} C   (datasheet-based hand calc {hc['tj_nosink']:.0f} C)")
    print(f"  Limit {TJ_LIMIT:.0f} C -> with sink {'PASS' if rs['tj'] <= TJ_LIMIT else 'FAIL'}, "
          f"no sink {'PASS' if rn['tj'] <= TJ_LIMIT else 'FAIL'}")

    with open("results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "hand_calc_Tj_C", "simulated_Tj_C", "Rjc_C_per_W", "Rcs_C_per_W", "Rsa_C_per_W"])
        w.writerow(["with_heatsink", f"{hc['tj_sink']:.1f}", f"{rs['tj']:.1f}",
                    f"{rs['r_jc']:.2f}", f"{rs['r_cs']:.2f}", f"{rs['r_sa']:.2f}"])
        w.writerow(["with_heatsink_hand_R", "", "", f"{hc['r_jc']:.2f}", f"{hc['r_cs']:.2f}", f"{hc['r_sa']:.2f}"])
        w.writerow(["no_heatsink", f"{hc['tj_nosink']:.0f}", f"{rn['tj']:.0f}", "", "", ""])
    print("\nFiles: assembly.stl  geometry.png  temperature.png  results.csv")


if __name__ == "__main__":
    main()
