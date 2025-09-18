
import streamlit as st
import numpy as np
import plotly.graph_objects as go

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

def plane_xy_bottom(limit=1.2, opacity=0.12):
    rng = np.linspace(-limit, limit, 2)
    X, Y = np.meshgrid(rng, rng)
    # XY plane at bottom face z = -limit
    s_xy = go.Surface(x=X, y=Y, z=np.full_like(X, -limit),
                      opacity=opacity, showscale=False, name='XY bottom')
    return s_xy

def planes_offset_zx_zy(limit=1.2, opacity=0.08, offset=0.3):
    rng = np.linspace(-limit, limit, 2)
    X, Y = np.meshgrid(rng, rng)
    # ZX plane at y=+offset
    s_zx = go.Surface(x=X, y=np.full_like(X, offset), z=Y,
                      opacity=opacity, showscale=False, name=f'ZX (y={offset})')
    # ZY plane at x=+offset
    s_zy = go.Surface(x=np.full_like(X, offset), y=X, z=Y,
                      opacity=opacity, showscale=False, name=f'ZY (x={offset})')
    return [s_zx, s_zy]

def projected_line_xy_bottom(start, vec, color='rgba(0,0,0,0.65)', width=6, limit=1.2):
    s=np.array(start); v=np.array(vec)
    tip = s + v
    eps = 1e-3
    z_plane = -limit + eps
    sP = np.array([s[0], s[1], z_plane])
    tP = np.array([tip[0], tip[1], z_plane])
    line=go.Scatter3d(x=[sP[0], tP[0]], y=[sP[1], tP[1]], z=[sP[2], tP[2]],
                      mode='lines+markers',
                      line=dict(color=color, width=width),
                      marker=dict(size=3, color=color),
                      name='proj', showlegend=False)
    return line

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Apparent Rotation Visualizer (3D + Bottom XY projection)", layout="wide")
st.title("Apparent Rotation Visualizer – 3D with XY projection on bottom face")

colL, colR = st.columns([1,1])
with colL:
    ang_yz = st.slider("YZ angulation α (about X) [deg]", -90.0, 90.0, 0.0, 1.0)
    ang_zx = st.slider("ZX angulation β (about Y) [deg]", -90.0, 90.0, 0.0, 1.0)
with colR:
    rot_yx = st.number_input("YX true rotation γ (about Z) [deg]", value=0.0, step=1.0, format="%.2f")
    offset  = st.slider("Anterior offset (units)", 0.0, 1.5, 0.0, 0.05)
    plane_offset = st.slider("ZX/ZY plane offset (away from origin)", 0.0, 1.0, 0.3, 0.05)

res = simulate_construct(ang_yz, ang_zx, rot_yx, L=1.0)

origin=np.zeros(3)
uZp=normalize(res["Z_prox"]); uZd=normalize(res["Z_dist"])
base_prox=origin+offset*uZp; base_dist=origin+offset*uZd

limit=1.2
traces = []
traces += axes_traces(limit=limit)
# planes: XY at bottom face, plus ZX & ZY offset away from origin
traces.append(plane_xy_bottom(limit=limit, opacity=0.12))
traces += planes_offset_zx_zy(limit=limit, opacity=0.08, offset=plane_offset)

# Main arrows
traces += arrow_trace(origin, res["Z_prox"],  'royalblue',  "Z_prox")
traces += arrow_trace(origin, res["Z_dist"],  'darkorange', "Z_dist")
traces += arrow_trace(base_prox, res["A_prox"], 'seagreen',  "A_prox")
traces += arrow_trace(base_dist, res["A_dist"], 'crimson',   "A_dist")

# Only XY projection ON THE BOTTOM FACE
traces += [projected_line_xy_bottom(origin,    res["Z_prox"], color='rgba(65,105,225,0.6)', limit=limit)]
traces += [projected_line_xy_bottom(origin,    res["Z_dist"], color='rgba(255,140,0,0.6)', limit=limit)]
traces += [projected_line_xy_bottom(base_prox, res["A_prox"], color='rgba(46,139,87,0.6)', limit=limit)]
traces += [projected_line_xy_bottom(base_dist, res["A_dist"], color='rgba(220,20,60,0.6)', limit=limit)]

fig3d = go.Figure(data=traces)
fig3d.update_layout(scene=dict(
    xaxis=dict(title='X', range=[-limit,limit]),
    yaxis=dict(title='Y', range=[-limit,limit]),
    zaxis=dict(title='Z', range=[-limit,limit]),
    aspectmode='cube'
), margin=dict(l=0,r=0,t=30,b=0))

st.plotly_chart(fig3d, use_container_width=True)

phi = res["phi_app"]
st.metric("φ apparent (deg)", f"{phi:.2f}" if np.isfinite(phi) else "undefined")
st.caption("Bottom face shows XY projection only; ZX and ZY planes are offset from the origin to declutter the view.")
