
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Core math utilities
# ---------------------------

def deg2rad(d):
    return np.deg2rad(d)

def rot_x(theta_deg):
    t = deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(theta_deg):
    t = deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(theta_deg):
    t = deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def angle_between_2d(u, v):
    ux, uy = u[:2]; vx, vy = v[:2]
    dot = ux*vx + uy*vy
    det = ux*vy - uy*vx
    return np.rad2deg(np.arctan2(det, dot))

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

# ---------------------------
# Geometry / simulation (body-fixed angulation default)
# ---------------------------

def simulate_construct(ang_yz_deg, ang_zx_deg, rot_yx_deg, L=1.0):
    # Proximal (reference)
    Z_prox_local = np.array([0.,0.,1.])  # +Z
    Y_prox_local = np.array([0.,1.,0.])  # +Y anterior

    # Distal local
    Z_dist_local = np.array([0.,0.,-1.])  # -Z
    Y_dist_local = np.array([0.,1.,0.])   # anterior

    # Body-fixed order: Y then X then Z
    R = rot_y(ang_zx_deg) @ rot_x(ang_yz_deg) @ rot_z(rot_yx_deg)

    Z_prox_global = Z_prox_local * L
    A_prox_global = Y_prox_local
    Z_dist_global = (R @ Z_dist_local) * L
    A_dist_global = R @ Y_dist_local

    # Apparent angle on XY
    Aprox_xy = A_prox_global.copy(); Aprox_xy[2] = 0.0
    Adist_xy = A_dist_global.copy(); Adist_xy[2] = 0.0
    if np.linalg.norm(Aprox_xy[:2]) < 1e-9 or np.linalg.norm(Adist_xy[:2]) < 1e-9:
        apparent_xy_deg = float('nan')
    else:
        apparent_xy_deg = angle_between_2d(normalize(Aprox_xy), normalize(Adist_xy))

    return {
        "Z_prox": Z_prox_global,
        "Z_dist": Z_dist_global,
        "A_prox": A_prox_global,
        "A_dist": A_dist_global,
        "phi_app": apparent_xy_deg,
    }

# ---------------------------
# Plot helpers
# ---------------------------

def plot_3d(res, offset=0.0):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    origin = np.zeros(3)

    col_Z_prox = 'tab:blue'
    col_Z_dist = 'tab:orange'
    col_A_prox = 'tab:green'
    col_A_dist = 'tab:red'
    axis_col = '0.6'

    # Offsets along Z
    uZp = normalize(res["Z_prox"])
    uZd = normalize(res["Z_dist"])
    base_prox = origin + offset * uZp
    base_dist = origin + offset * uZd

    # Main axes
    ax.quiver(*origin, *res["Z_prox"], length=1.0, arrow_length_ratio=0.1,
              color=col_Z_prox, linewidths=4)
    ax.text(*(res["Z_prox"]*1.1), "Z_prox", color=col_Z_prox)
    ax.quiver(*origin, *res["Z_dist"], length=1.0, arrow_length_ratio=0.1,
              color=col_Z_dist, linewidths=4)
    ax.text(*(res["Z_dist"]*1.1), "Z_dist", color=col_Z_dist)

    # Anterior
    ax.quiver(*base_prox, *res["A_prox"], length=1.0, arrow_length_ratio=0.1,
              color=col_A_prox, linewidths=4)
    ax.text(*(base_prox+res["A_prox"]*1.1), "A_prox", color=col_A_prox)
    ax.quiver(*base_dist, *res["A_dist"], length=1.0, arrow_length_ratio=0.1,
              color=col_A_dist, linewidths=4)
    ax.text(*(base_dist+res["A_dist"]*1.1), "A_dist", color=col_A_dist)

    # World axes
    axis_lim = 1.2
    ax.plot([0, axis_lim],[0,0],[0,0],color=axis_col); ax.text(axis_lim,0,0,'X',color=axis_col)
    ax.plot([0,0],[0,axis_lim],[0,0],color=axis_col); ax.text(0,axis_lim,0,'Y',color=axis_col)
    ax.plot([0,0],[0,0],[0,axis_lim],color=axis_col); ax.text(0,0,axis_lim,'Z',color=axis_col)
    ax.set_xlim(-axis_lim,axis_lim); ax.set_ylim(-axis_lim,axis_lim); ax.set_zlim(-axis_lim,axis_lim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    return fig

def plot_xy(res):
    fig, ax = plt.subplots(figsize=(5,5))
    Aprox = res["A_prox"].copy(); Aprox[2]=0
    Adist = res["A_dist"].copy(); Adist[2]=0
    ax.arrow(0,0,Aprox[0],Aprox[1],width=0.04,head_width=0.12,
             length_includes_head=True,color='tab:green')
    ax.arrow(0,0,Adist[0],Adist[1],width=0.04,head_width=0.12,
             length_includes_head=True,color='tab:red')
    ax.set_aspect('equal'); ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    phi = res["phi_app"]
    subtitle = f"φ apparent = {phi:.2f}°" if np.isfinite(phi) else "undefined"
    ax.set_title(f"XY projection\n{subtitle}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    return fig

def plot_zx(res):
    fig, ax = plt.subplots(figsize=(3.5,3))
    Aprox=[res["A_prox"][0],res["A_prox"][2]]
    Adist=[res["A_dist"][0],res["A_dist"][2]]
    Zprox=[res["Z_prox"][0],res["Z_prox"][2]]
    Zdist=[res["Z_dist"][0],res["Z_dist"][2]]
    for vec,col in [(Aprox,'tab:green'),(Adist,'tab:red'),(Zprox,'tab:blue'),(Zdist,'tab:orange')]:
        ax.arrow(0,0,vec[0],vec[1],width=0.04,head_width=0.12,
                 length_includes_head=True,color=col)
    ax.set_aspect('equal'); ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_title("ZX (Z vertical)")
    return fig

def plot_zy(res):
    fig, ax = plt.subplots(figsize=(3.5,3))
    Aprox=[res["A_prox"][1],res["A_prox"][2]]
    Adist=[res["A_dist"][1],res["A_dist"][2]]
    Zprox=[res["Z_prox"][1],res["Z_prox"][2]]
    Zdist=[res["Z_dist"][1],res["Z_dist"][2]]
    for vec,col in [(Aprox,'tab:green'),(Adist,'tab:red'),(Zprox,'tab:blue'),(Zdist,'tab:orange')]:
        ax.arrow(0,0,vec[0],vec[1],width=0.04,head_width=0.12,
                 length_includes_head=True,color=col)
    ax.set_aspect('equal'); ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    ax.set_xlabel("Y"); ax.set_ylabel("Z"); ax.set_title("ZY (Z vertical)")
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Apparent Rotation Visualizer", layout="wide")

st.title("Apparent Rotation Visualizer")
st.markdown(
    "Adjust sagittal (YZ) and coronal (ZX) angulation and see how apparent "
    "rotation shows up on axial CT, even if true torsion (YX) is fixed.\n\n"
    "Mode: **body-fixed angulation**."
)

col1,col2 = st.columns([1,1])

with col1:
    ang_yz = st.slider("YZ angulation α (about X)", -90.0, 90.0, 0.0, 1.0)
    ang_zx = st.slider("ZX angulation β (about Y)", -90.0, 90.0, 0.0, 1.0)
    rot_yx = st.number_input("YX true rotation γ (about Z)", value=0.0, step=1.0, format="%.2f")
    offset = st.slider("Anterior offset", 0.0, 1.5, 0.0, 0.05)

res = simulate_construct(ang_yz, ang_zx, rot_yx, L=1.0)

with col2:
    phi = res["phi_app"]
    if np.isfinite(phi):
        st.metric("φ apparent (deg)", f"{phi:.2f}")
    else:
        st.info("Apparent angle undefined (vector projects near zero).")

c1,c2,c3 = st.columns([1.2,1.0,0.8])
with c1:
    st.pyplot(plot_3d(res, offset=offset), clear_figure=True)
with c2:
    st.pyplot(plot_xy(res), clear_figure=True)
with c3:
    st.pyplot(plot_zx(res), clear_figure=True)
    st.pyplot(plot_zy(res), clear_figure=True)

st.caption("Formula: φ_app = atan2(sinγ·cosβ − cosγ·sinα·sinβ , cosγ·cosα)")
