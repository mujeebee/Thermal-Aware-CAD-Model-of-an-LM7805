"""
Design sweeps: (1) sensitivity to the convection coefficient h on YOUR NX geometry,
(2) effect of fin count.  Writes sweeps.png and prints the tables.

Run (same folder as regulator_heatsink.py and solve_from_stl.py):
    python sweeps.py pcb.stl 7805_regulator.stl heatsink.stl
(order: PCB, regulator, heat sink)
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regulator_heatsink as rh
import solve_from_stl as ss

# ---------- 1. h sensitivity on the user's NX geometry (STL) ----------
if len(sys.argv)!=4: sys.exit("Usage: python sweeps.py pcb.stl regulator.stl heatsink.stl")
solids=[ss.read_stl(p)[0] for p in sys.argv[1:4]]
allv=np.concatenate([s.reshape(-1,3) for s in solids]); lo,hi=allv.min(0),allv.max(0)
R=rh.RES; origin=np.floor(lo/R)*R-1.0; shape=tuple(int(np.ceil((hi[i]-origin[i])/R))+2 for i in range(3))
mat=np.zeros(shape,np.int8)
for code,t in zip([rh.FR4,rh.PKGM,rh.AL],solids):
    f,_=ss.voxelize(t,origin,shape); mat[f&(mat==0)]=code
    if code==rh.PKGM: mat[f]=code
xc,yc,zc=[origin[i]+(np.arange(shape[i])+.5)*R for i in range(3)]
pk=np.argwhere(mat==rh.PKGM); x_c=xc[pk[:,0]].mean(); z_c=zc[pk[:,2]].mean(); y_top=yc[pk[:,1]].max()+R/2
X,Y,Z=np.meshgrid(xc,yc,zc,indexing="ij")
die=((abs(X-x_c)<=1.5)&(Y>=y_top-2.75)&(Y<y_top-1.75)&(abs(Z-z_c)<=1.5)&(mat==rh.PKGM))
hs=[5,7.5,10,12.5,15]; sim_h=[]; hand_h=[]
for h in hs:
    rh.H_CONV=h
    sim_h.append(rh.simulate(True,voxels=(mat,die,(xc,yc,zc)))["tj"]); hand_h.append(rh.hand_calc()["tj_sink"])
rh.H_CONV=10.0
print("h sensitivity (NX geometry):"); 
for h,s,hd in zip(hs,sim_h,hand_h): print(f"  h={h:5.1f}  sim Tj={s:6.1f}  hand Tj={hd:6.1f}")

# ---------- 2. fin-count sweep (built-in parametric geometry) ----------
def set_fins(n):
    rh.N_FINS=n; rh.PITCH=(rh.SINK_W-rh.FIN_T)/(n-1)
    rh.FINS=[dict(x=(rh.SINK_X0+i*rh.PITCH,rh.SINK_X0+i*rh.PITCH+rh.FIN_T),
        y=(rh.SINK_Y0+rh.SINK_BASE_T,rh.SINK_Y0+rh.SINK_BASE_T+rh.FIN_H),z=(rh.SINK_Z0,rh.SINK_Z1)) for i in range(n)]
    rh.FIN_GAP=rh.PITCH-rh.FIN_T
ns=[3,4,5,6,7,8,9]; sim_n=[]; hand_n=[]; gaps=[]
for n in ns:
    set_fins(n); sim_n.append(rh.simulate(True)["tj"]); hand_n.append(rh.hand_calc()["tj_sink"]); gaps.append(rh.FIN_GAP)
print("\nfin-count sweep (h=10):")
for n,s,hd,g in zip(ns,sim_n,hand_n,gaps): print(f"  fins={n}  gap={g:4.1f} mm  sim Tj={s:6.1f}  hand Tj={hd:6.1f}")

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(hs,sim_h,"o-",label="Simulation (your NX geometry)"); ax[0].plot(hs,hand_h,"s--",label="Hand calculation")
ax[0].axhline(100,color="r",ls=":",label="100 °C design limit"); ax[0].set_xlabel("Convection coefficient h (W/m²K)")
ax[0].set_ylabel("Junction temperature (°C)"); ax[0].set_title("Sensitivity to the assumed h"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ns,sim_n,"o-",label="Simulation"); ax[1].plot(ns,hand_n,"s--",label="Hand calculation")
ax[1].axhline(100,color="r",ls=":",label="100 °C design limit"); ax[1].set_xlabel("Number of fins")
ax[1].set_ylabel("Junction temperature (°C)"); ax[1].set_title("Effect of fin count (h = 10 W/m²K)"); ax[1].legend(); ax[1].grid(alpha=.3)
for n,g,s in zip(ns,gaps,sim_n): ax[1].annotate(f"{g:.1f} mm",(n,s),textcoords="offset points",xytext=(0,8),ha="center",fontsize=8)
fig.tight_layout(); fig.savefig("sweeps.png",dpi=150)
