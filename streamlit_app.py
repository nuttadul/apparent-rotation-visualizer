
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
    return np.rad2deg(np.arctan2(det, dot))
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

# ---------- Plotly helpers ----------
def arrow_trace(start, vec, color, name):
    s=np.array(start); v=np.array(vec)
    tip=s+v
    line=go.Scatter3d(x=[s[0],tip[0]], y=[s[1],tip[1]], z=[s[2],tip[2]],
                      mode='lines', line=dict(color=color, width=8), name=name, showlegend=True)
    # cone size scales to vector length
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

def plane_surfaces(limit=1.2, opacity=0.1):
    # Draw XY (z=0), ZX (y=0), ZY (x=0) translucent planes
    rng = np.linspace(-limit, limit, 2)
    X, Y = np.meshgrid(rng, rng)
    # XY plane z=0
    s_xy = go.Surface(x=X, y=Y, z=np.zeros_like(X), opacity=opacity, showscale=False, name='XY plane')
    # ZX plane y=0 (plot with x,z)
    s_zx = go.Surface(x=X, y=np.zeros_like(X), z=Y, opacity=opacity, showscale=False, name='ZX plane')
    # ZY plane x=0 (plot with y,z)
    s_zy = go.Surface(x=np.zeros_like(X), y=X, z=Y, opacity=opacity, showscale=False, name='ZY plane')
    return [s_xy, s_zx, s_zy]

def projected_line(start, vec, plane='xy', color='rgba(0,0,0,0.6)', name='proj', width=6):
    # Orthogonal projection of an arrow onto a coordinate plane
    s=np.array(start); v=np.array(vec)
    tip = s + v
    eps = 1e-3
    if plane=='xy':
        sP = np.array([s[0], s[1], 0.0 + eps])
        tP = np.array([tip[0], tip[1], 0.0 + eps])
    elif plane=='zx':  # y=0
        sP = np.array([s[0], 0.0 + eps, s[2]])
        tP = np.array([tip[0], 0.0 + eps, tip[2]])
    elif plane=='zy':  # x=0
        sP = np.array([0.0 + eps, s[1], s[2]])
        tP = np.array([0.0 + eps, tip[1], tip[2]])
    else:
        raise ValueError("plane must be 'xy', 'zx', or 'zy'")
    line=go.Scatter3d(x=[sP[0], tP[0]], y=[sP[1], tP[1]], z=[sP[2], tP[2]],
                      mode='lines+markers',
                      line=dict(color=color, width=width),
                      marker=dict(size=3, color=color),
                      name=name, showlegend=False)
    return line

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
st.set_page_config(page_title="Apparent Rotation Visualizer (Interactive 3D + Projections)", layout="wide")
st.title("Apparent Rotation Visualizer – Interactive 3D + On-cube Projections")

colL, colR = st.columns([1,1])
with colL:
    ang_yz = st.slider("YZ angulation α (about X) [deg]", -90.0, 90.0, 0.0, 1.0)
    ang_zx = st.slider("ZX angulation β (about Y) [deg]", -90.0, 90.0, 0.0, 1.0)
with colR:
    rot_yx = st.number_input("YX true rotation γ (about Z) [deg]", value=0.0, step=1.0, format="%.2f")
    offset  = st.slider("Anterior offset (units)", 0.0, 1.5, 0.0, 0.05)
    show_proj = st.checkbox("Show projections on XY / ZX / ZY planes", value=True)

res = simulate_construct(ang_yz, ang_zx, rot_yx, L=1.0)

# --- Plotly 3D with planes and projections ---
origin=np.zeros(3)
uZp=normalize(res["Z_prox"]); uZd=normalize(res["Z_dist"])
base_prox=origin+offset*uZp; base_dist=origin+offset*uZd

traces = []
traces += axes_traces(limit=1.2)
traces += plane_surfaces(limit=1.2, opacity=0.10 if show_proj else 0.0)

# Main arrows
traces += arrow_trace(origin, res["Z_prox"],  'royalblue',  "Z_prox")
traces += arrow_trace(origin, res["Z_dist"],  'darkorange', "Z_dist")
traces += arrow_trace(base_prox, res["A_prox"], 'seagreen',  "A_prox")
traces += arrow_trace(base_dist, res["A_dist"], 'crimson',   "A_dist")

# Projections (lines only, semi-transparent)
if show_proj:
    fade = 0.7
    traces += [projected_line(origin, res["Z_prox"], plane='xy', color='rgba(65,105,225,0.6)', name='Z_prox→XY')]
    traces += [projected_line(origin, res["Z_dist"], plane='xy', color='rgba(255,140,0,0.6)', name='Z_dist→XY')]
    traces += [projected_line(base_prox, res["A_prox"], plane='xy', color='rgba(46,139,87,0.6)', name='A_prox→XY')]
    traces += [projected_line(base_dist, res["A_dist"], plane='xy', color='rgba(220,20,60,0.6)', name='A_dist→XY')]

    traces += [projected_line(origin, res["Z_prox"], plane='zx', color='rgba(65,105,225,0.6)', name='Z_prox→ZX')]
    traces += [projected_line(origin, res["Z_dist"], plane='zx', color='rgba(255,140,0,0.6)', name='Z_dist→ZX')]
    traces += [projected_line(base_prox, res["A_prox"], plane='zx', color='rgba(46,139,87,0.6)', name='A_prox→ZX')]
    traces += [projected_line(base_dist, res["A_dist"], plane='zx', color='rgba(220,20,60,0.6)', name='A_dist→ZX')]

    traces += [projected_line(origin, res["Z_prox"], plane='zy', color='rgba(65,105,225,0.6)', name='Z_prox→ZY')]
    traces += [projected_line(origin, res["Z_dist"], plane='zy', color='rgba(255,140,0,0.6)', name='Z_dist→ZY')]
    traces += [projected_line(base_prox, res["A_prox"], plane='zy', color='rgba(46,139,87,0.6)', name='A_prox→ZY')]
    traces += [projected_line(base_dist, res["A_dist"], plane='zy', color='rgba(220,20,60,0.6)', name='A_dist→ZY')]

fig3d = go.Figure(data=traces)
fig3d.update_layout(scene=dict(
    xaxis=dict(title='X', range=[-1.2,1.2], backgroundcolor="rgba(0,0,0,0)"),
    yaxis=dict(title='Y', range=[-1.2,1.2], backgroundcolor="rgba(0,0,0,0)"),
    zaxis=dict(title='Z', range=[-1.2,1.2], backgroundcolor="rgba(0,0,0,0)"),
    aspectmode='cube'
), margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=0.95))

st.plotly_chart(fig3d, use_container_width=True)

# --- 2D projections below for precise reading ---
c1,c2,c3 = st.columns([1.0,0.9,0.9])
with c1: st.pyplot(plot_xy_mat(res), clear_figure=True)
with c2: st.pyplot(plot_plane_mat(res,'ZX'), clear_figure=True)
with c3: st.pyplot(plot_plane_mat(res,'ZY'), clear_figure=True)

phi = res["phi_app"]
st.metric("φ apparent (deg)", f"{phi:.2f}" if np.isfinite(phi) else "undefined")
st.caption("3D shows vectors + their orthogonal projections on XY (z=0), ZX (y=0), and ZY (x=0) planes. "
           "Formula: φ_app = atan2(sinγ·cosβ − cosγ·sinα·sinβ , cosγ·cosα).")
