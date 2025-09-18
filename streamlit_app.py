
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------- Math utilities ----------
def deg2rad(d): return np.deg2rad(d)
def rot_x(tdeg):
    t=deg2rad(tdeg); c,s=np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(tdeg):
    t=deg2rad(tdeg); c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(tdeg):
    t=deg2rad(tdeg); c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def angle_between_2d(u,v):
    ux,uy=u[:2]; vx,vy=v[:2]
    dot=ux*vx+uy*vy; det=ux*vy-uy*vx
    return np.rad2deg(np.arctan2(det,dot))
def normalize(v):
    n=np.linalg.norm(v); 
    return v if n==0 else v/n

# ---------- Simulation (body-fixed) ----------
def simulate_construct(ang_yz_deg, ang_zx_deg, rot_yx_deg, L=1.0):
    Z_prox_local=np.array([0.,0.,1.]); Y_prox_local=np.array([0.,1.,0.])
    Z_dist_local=np.array([0.,0.,-1.]); Y_dist_local=np.array([0.,1.,0.])
    R = rot_y(ang_zx_deg) @ rot_x(ang_yz_deg) @ rot_z(rot_yx_deg)
    Z_prox = Z_prox_local*L; A_prox = Y_prox_local
    Z_dist = (R@Z_dist_local)*L; A_dist = R@Y_dist_local
    Aprox_xy=A_prox.copy(); Aprox_xy[2]=0.0
    Adist_xy=A_dist.copy(); Adist_xy[2]=0.0
    if np.linalg.norm(Aprox_xy[:2])<1e-9 or np.linalg.norm(Adist_xy[:2])<1e-9:
        phi=float('nan')
    else:
        phi=angle_between_2d(normalize(Aprox_xy), normalize(Adist_xy))
    return {"Z_prox":Z_prox,"Z_dist":Z_dist,"A_prox":A_prox,"A_dist":A_dist,"phi_app":phi}

# ---------- Plotly 3D arrows ----------
def arrow_trace(start, vec, color, name):
    s=np.array(start); v=np.array(vec)
    tip=s+v
    line=go.Scatter3d(x=[s[0],tip[0]], y=[s[1],tip[1]], z=[s[2],tip[2]],
                      mode='lines', line=dict(color=color, width=8), name=name, showlegend=True)
    ln=np.linalg.norm(v); size=max(0.08, 0.15*ln)
    cone=go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]],
                 u=[v[0]], v=[v[1]], w=[v[2]], sizemode='absolute', sizeref=size,
                 showscale=False, colorscale=[[0, color],[1, color]], anchor='tip', name=name)
    return [line, cone]

def axes_traces(limit=1.2):
    x=go.Scatter3d(x=[0,limit], y=[0,0], z=[0,0], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False)
    y=go.Scatter3d(x=[0,0], y=[0,limit], z=[0,0], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False)
    z=go.Scatter3d(x=[0,0], y=[0,0], z=[0,limit], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False)
    return [x,y,z]

# ---------- Matplotlib 2D helpers ----------
def plot_xy_mat(res):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    Aprox=res["A_prox"].copy(); Aprox[2]=0
    Adist=res["A_dist"].copy(); Adist[2]=0
    ax.arrow(0,0,Aprox[0],Aprox[1],width=0.04,head_width=0.12,length_includes_head=True,color='tab:green')
    ax.arrow(0,0,Adist[0],Adist[1],width=0.04,head_width=0.12,length_includes_head=True,color='tab:red')
    ax.set_aspect('equal'); ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    phi=res["phi_app"]; txt=f"φ apparent = {phi:.2f}°" if np.isfinite(phi) else "undefined"
    ax.set_title(f"XY projection\n{txt}"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    return fig

def plot_plane_mat(res, plane='ZX'):
    fig, ax = plt.subplots(figsize=(3.5,3))
    if plane=='ZX':
        Aprox=[res["A_prox"][0],res["A_prox"][2]]; Adist=[res["A_dist"][0],res["A_dist"][2]]
        Zprox=[res["Z_prox"][0],res["Z_prox"][2]]; Zdist=[res["Z_dist"][0],res["Z_dist"][2]]
        ax.set_xlabel("X"); ax.set_ylabel("Z"); title="ZX (Z vertical)"
    else:
        Aprox=[res["A_prox"][1],res["A_prox"][2]]; Adist=[res["A_dist"][1],res["A_dist"][2]]
        Zprox=[res["Z_prox"][1],res["Z_prox"][2]]; Zdist=[res["Z_dist"][1],res["Z_dist"][2]]
        ax.set_xlabel("Y"); ax.set_ylabel("Z"); title="ZY (Z vertical)"
    for vec,col in [(Aprox,'tab:green'),(Adist,'tab:red'),(Zprox,'tab:blue'),(Zdist,'tab:orange')]:
        ax.arrow(0,0,vec[0],vec[1],width=0.04,head_width=0.12,length_includes_head=True,color=col)
    ax.set_aspect('equal'); ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2); ax.set_title(title)
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Apparent Rotation Visualizer (Interactive 3D)", layout="wide")
st.title("Apparent Rotation Visualizer – Interactive 3D")

colL, colR = st.columns([1,1])
with colL:
    ang_yz = st.slider("YZ angulation α (about X) [deg]", -90.0, 90.0, 0.0, 1.0)
    ang_zx = st.slider("ZX angulation β (about Y) [deg]", -90.0, 90.0, 0.0, 1.0)
with colR:
    rot_yx = st.number_input("YX true rotation γ (about Z) [deg]", value=0.0, step=1.0, format="%.2f")
    offset  = st.slider("Anterior offset (units)", 0.0, 1.5, 0.0, 0.05)

res = simulate_construct(ang_yz, ang_zx, rot_yx, L=1.0)

# --- Plotly 3D ---
origin=np.zeros(3)
uZp=normalize(res["Z_prox"]); uZd=normalize(res["Z_dist"])
base_prox=origin+offset*uZp; base_dist=origin+offset*uZd

traces = []
traces += axes_traces(limit=1.2)
traces += arrow_trace(origin, res["Z_prox"],  'royalblue',  "Z_prox")
traces += arrow_trace(origin, res["Z_dist"],  'darkorange', "Z_dist")
traces += arrow_trace(base_prox, res["A_prox"], 'seagreen',  "A_prox")
traces += arrow_trace(base_dist, res["A_dist"], 'crimson',   "A_dist")

fig3d = go.Figure(data=traces)
fig3d.update_layout(scene=dict(
    xaxis=dict(title='X', range=[-1.2,1.2]),
    yaxis=dict(title='Y', range=[-1.2,1.2]),
    zaxis=dict(title='Z', range=[-1.2,1.2]),
    aspectmode='cube'
), margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=0.95))

st.plotly_chart(fig3d, use_container_width=True)

# --- 2D projections ---
c1,c2,c3 = st.columns([1.0,0.9,0.9])
with c1: st.pyplot(plot_xy_mat(res), clear_figure=True)
with c2: st.pyplot(plot_plane_mat(res,'ZX'), clear_figure=True)
with c3: st.pyplot(plot_plane_mat(res,'ZY'), clear_figure=True)

phi = res["phi_app"]
st.metric("φ apparent (deg)", f"{phi:.2f}" if np.isfinite(phi) else "undefined")
st.caption("3D view is draggable/zoomable (Plotly). Formula: φ_app = atan2(sinγ·cosβ − cosγ·sinα·sinβ , cosγ·cosα).")
