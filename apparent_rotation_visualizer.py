import numpy as np
import matplotlib as mpl
# Try to ensure an interactive backend if a non-interactive one (e.g., Agg) is active
try:
    if str(mpl.get_backend()).lower() == 'agg':
        mpl.use('TkAgg')  # requires Tk installed
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from dataclasses import dataclass
import argparse
import sys
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox

# ---------------------------
# Core math utilities
# ---------------------------

def deg2rad(d): return np.deg2rad(d)

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
    n = np.linalg.norm(v);
    return v if n == 0 else v/n

# ---------------------------
# Geometry
# ---------------------------

@dataclass
class DeformInput:
    ang_yz_deg: float   # about X (YZ-plane angulation)
    ang_zx_deg: float   # about Y (ZX-plane angulation)
    rot_yx_deg: float   # about Z (YX-plane rotation)

@dataclass
class ConstructResult:
    A_prox_global: np.ndarray
    A_dist_global: np.ndarray
    Z_prox_global: np.ndarray
    Z_dist_global: np.ndarray
    apparent_xy_deg: float
    R_dist: np.ndarray

def simulate_construct(inputs, L_prox=1.0, L_dist=1.0, world_fixed_angulation=False):
    """
    Two connected segments share the interconnection at the origin.
    Initially the proximal main vector is +Z and the distal main vector is −Z (colinear, opposite) and 'anterior' lies along +Y.
    Distal is rotated relative to proximal by:
        1) about X (YZ-plane angulation),
        2) about Y (ZX-plane angulation),
        3) about Z (YX-plane rotation).
    """
    # Proximal reference
    Z_prox_local = np.array([0.,0.,1.])
    Y_prox_local = np.array([0.,1.,0.])  # anterior

    Z_dist_local = np.array([0., 0., -1.])
    Y_dist_local = np.array([0., 1., 0.])  # anterior stays +Y

    # Global (proximal kept as reference)
    Z_prox_global = Z_prox_local * L_prox
    A_prox_global = Y_prox_local

    if world_fixed_angulation:
        # World-fixed X/Y angulation (EXTRINSIC): apply X then Y about global axes, then axial Z last
        R_ang = rot_y(inputs.ang_zx_deg) @ rot_x(inputs.ang_yz_deg)
        R = rot_z(inputs.rot_yx_deg) @ R_ang
    else:
        # Body-fixed X/Y angulation (INTRINSIC relative to already-rotated frame):
        # apply axial Z first, then X and Y about the rotated axes
        R = rot_y(inputs.ang_zx_deg) @ rot_x(inputs.ang_yz_deg) @ rot_z(inputs.rot_yx_deg)

    # Apply to distal
    Z_dist_global = (R @ Z_dist_local) * L_dist
    A_dist_global = R @ Y_dist_local

    # Measure apparent axial (XY) angle between anterior directions
    Aprox_xy = A_prox_global.copy(); Aprox_xy[2] = 0.
    Adist_xy = A_dist_global.copy(); Adist_xy[2] = 0.

    if np.linalg.norm(Aprox_xy[:2]) < 1e-9 or np.linalg.norm(Adist_xy[:2]) < 1e-9:
        apparent_xy_deg = float('nan')
    else:
        apparent_xy_deg = angle_between_2d(normalize(Aprox_xy), normalize(Adist_xy))

    return ConstructResult(A_prox_global, A_dist_global, Z_prox_global, Z_dist_global, apparent_xy_deg, R)

# ---------------------------
# Visualization
# ---------------------------

def draw_3d_construct(res, title='3D Construct'):
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    origin = np.zeros(3)
    # Main axes (Z) as arrows
    ax.quiver(*origin, *res.Z_prox_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
    ax.text(*(res.Z_prox_global * 1.1), "Z_prox")
    ax.quiver(*origin, *res.Z_dist_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
    ax.text(*(res.Z_dist_global * 1.1), "Z_dist")
    # Anterior (+Y) as arrows
    ax.quiver(*origin, *res.A_prox_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
    ax.text(*(res.A_prox_global * 1.1), "A_prox")
    ax.quiver(*origin, *res.A_dist_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
    ax.text(*(res.A_dist_global * 1.1), "A_dist")
    # Axes
    axis_lim = 1.2
    ax.plot([0, axis_lim], [0, 0], [0, 0]); ax.text(axis_lim,0,0,'X')
    ax.plot([0, 0], [0, axis_lim], [0, 0]); ax.text(0,axis_lim,0,'Y')
    ax.plot([0, 0], [0, 0], [0, axis_lim]); ax.text(0,0,axis_lim,'Z')
    ax.set_xlim(-axis_lim, axis_lim); ax.set_ylim(-axis_lim, axis_lim); ax.set_zlim(-axis_lim, axis_lim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def draw_xy_projection(res, title_prefix='XY Projection (Anterior vectors)'):
    fig = plt.figure(); ax = fig.add_subplot(111)
    # Project anterior vectors onto XY
    Aprox_xy = res.A_prox_global.copy(); Aprox_xy[2] = 0.
    Adist_xy = res.A_dist_global.copy(); Adist_xy[2] = 0.
    # Unit circle for reference
    circle = plt.Circle((0,0), 1.0, fill=False); ax.add_artist(circle)
    # Arrows
    ax.arrow(0, 0, Aprox_xy[0], Aprox_xy[1], width=0.04, head_width=0.12, length_includes_head=True)
    ax.arrow(0, 0, Adist_xy[0], Adist_xy[1], width=0.04, head_width=0.12, length_includes_head=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    subtitle = f"Apparent axial angle = {res.apparent_xy_deg:.2f}°" if np.isfinite(res.apparent_xy_deg) else "Apparent axial angle = undefined"
    ax.set_title(f"{title_prefix}\n{subtitle}")
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    plt.show()

def draw_zx_projection(res, title_prefix='ZX Projection (Main axes)'):
    fig = plt.figure(); ax = fig.add_subplot(111)
    # Project main axes (Z) into ZX plane
    Zprox_zx = np.array([res.Z_prox_global[2], res.Z_prox_global[0]])
    Zdist_zx = np.array([res.Z_dist_global[2], res.Z_dist_global[0]])
    ax.arrow(0, 0, Zprox_zx[0], Zprox_zx[1], width=0.04, head_width=0.12, length_includes_head=True)
    ax.arrow(0, 0, Zdist_zx[0], Zdist_zx[1], width=0.04, head_width=0.12, length_includes_head=True)
    ax.set_aspect('equal', adjustable='box')
    lim = 1.2; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Z'); ax.set_ylabel('X'); ax.set_title(title_prefix)
    plt.show()

def run_animation(ang_yz_deg_target, ang_zx_deg_target, rot_yx_deg_fixed, L=1.0, frames=60):
    """
    Animate apparent axial rotation while true axial rotation (about Z) is fixed.
    Both angulations about X (YZ-plane) and about Y (ZX-plane) increase linearly from 0 to their targets.
    A slider lets you scrub the progression manually; a Play/Pause button controls auto-play.
    """
    # Build figure with two axes: 3D construct and XY projection
    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(121, projection='3d')
    axxy = fig.add_subplot(122)

    # Space for slider and button
    plt.subplots_adjust(bottom=0.22)
    ax_slider = plt.axes([0.12, 0.08, 0.65, 0.03])
    ax_btn = plt.axes([0.80, 0.06, 0.12, 0.05])

    # Slider from 0..1 (fraction of target angulation)
    s_frac = Slider(ax_slider, "Angulation fraction", 0.0, 1.0, valinit=0.0)
    btn = Button(ax_btn, "Play")

    playing = {"state": False}

    def render(tfrac):
        # Compute current angles by linear interpolation (0 -> target), axial rot fixed
        ang_yz = tfrac * ang_yz_deg_target
        ang_zx = tfrac * ang_zx_deg_target
        inputs = DeformInput(ang_yz_deg=ang_yz, ang_zx_deg=ang_zx, rot_yx_deg=rot_yx_deg_fixed)
        res = simulate_construct(inputs, L, L)

        # --- draw 3D ---
        ax3d.cla()
        origin = np.zeros(3)
        ax3d.quiver(*origin, *res.Z_prox_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
        ax3d.text(*(res.Z_prox_global * 1.1), "Z_prox")
        ax3d.quiver(*origin, *res.Z_dist_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
        ax3d.text(*(res.Z_dist_global * 1.1), "Z_dist")
        ax3d.quiver(*origin, *res.A_prox_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
        ax3d.text(*(res.A_prox_global * 1.1), "A_prox")
        ax3d.quiver(*origin, *res.A_dist_global, length=1.0, normalize=False, arrow_length_ratio=0.1, linewidths=4)
        ax3d.text(*(res.A_dist_global * 1.1), "A_dist")
        axis_lim = 1.2
        ax3d.plot([0, axis_lim], [0, 0], [0, 0]); ax3d.text(axis_lim,0,0,'X')
        ax3d.plot([0, 0], [0, axis_lim], [0, 0]); ax3d.text(0,axis_lim,0,'Y')
        ax3d.plot([0, 0], [0, 0], [0, axis_lim]); ax3d.text(0,0,axis_lim,'Z')
        ax3d.set_xlim(-axis_lim, axis_lim); ax3d.set_ylim(-axis_lim, axis_lim); ax3d.set_zlim(-axis_lim, axis_lim)
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        ax3d.set_title(f"3D (YZ={ang_yz:.1f}°, ZX={ang_zx:.1f}°, YX={rot_yx_deg_fixed:.1f}°)")

        # --- draw XY projection ---
        axxy.cla()
        Aprox_xy = res.A_prox_global.copy(); Aprox_xy[2] = 0.0
        Adist_xy = res.A_dist_global.copy(); Adist_xy[2] = 0.0
        circle = plt.Circle((0,0), 1.0, fill=False); axxy.add_artist(circle)
        axxy.arrow(0, 0, Aprox_xy[0], Aprox_xy[1], width=0.04, head_width=0.12, length_includes_head=True)
        axxy.arrow(0, 0, Adist_xy[0], Adist_xy[1], width=0.04, head_width=0.12, length_includes_head=True)
        axxy.set_aspect('equal', adjustable='box')
        axxy.set_xlim(-1.2, 1.2); axxy.set_ylim(-1.2, 1.2)
        subtitle = f"Apparent axial angle = {res.apparent_xy_deg:.2f}°"
        axxy.set_title(f"XY Projection\n{subtitle}")
        axxy.set_xlabel("X"); axxy.set_ylabel("Y")
        fig.canvas.draw_idle()

    # Initialize
    render(0.0)

    # Slider callback
    def on_slide(val):
        render(s_frac.val)
    s_frac.on_changed(on_slide)

    # Animation function
    def make_anim():
        return animation.FuncAnimation(
            fig,
            lambda i: (s_frac.set_val(i / max(1, frames-1)),),
            frames=frames,
            interval=60,
            blit=False,
            repeat=False
        )

    anim_container = {"anim": None}

    # Play/Pause button
    def on_click(event):
        if not playing["state"]:
            playing["state"] = True
            btn.label.set_text("Pause")
            anim_container["anim"] = make_anim()
        else:
            # Pause by toggling state; rebuilding anim on resume
            playing["state"] = False
            btn.label.set_text("Play")
            # No direct pause in FuncAnimation; leaving as is stops progression when button toggled before completion.
    btn.on_clicked(on_click)

    plt.show()

def run_panel(default_rot_yx: float = 0.0, L: float = 1.0):
    """
    Interactive panel:
    - Sliders: YZ angulation (about X), ZX angulation (about Y)
    - Text box: YX true rotation (about Z)
    Shows: 3D construct (left) and XY projection (right), updated live.
    """
    # Figure layout with side column of two smaller projections (ZX on top, ZY on bottom)
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1.0, 0.6])
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    axxy = fig.add_subplot(gs[0, 1])
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.35)
    axzx = fig.add_subplot(gs_right[0, 0])
    axzy = fig.add_subplot(gs_right[1, 0])

    # Space at the bottom for controls
    plt.subplots_adjust(bottom=0.28)

    # Controls: two sliders + offset slider + rotation TextBox
    ax_s_yz = plt.axes([0.10, 0.16, 0.78, 0.03])
    ax_s_zx = plt.axes([0.10, 0.11, 0.78, 0.03])
    ax_tb_rot = plt.axes([0.10, 0.05, 0.18, 0.05])
    ax_btn_reset = plt.axes([0.32, 0.05, 0.12, 0.05])
    ax_s_off = plt.axes([0.10, 0.21, 0.78, 0.03])

    s_yz = Slider(ax_s_yz, "YZ ang (°) about X", -90.0, 90.0, valinit=0.0)
    s_zx = Slider(ax_s_zx, "ZX ang (°) about Y", -90.0, 90.0, valinit=0.0)
    s_off = Slider(ax_s_off, "Anterior offset (units)", 0.0, 1.5, valinit=0.0)
    tb_rot = TextBox(ax_tb_rot, "YX rot (°) about Z", initial=f"{default_rot_yx:.2f}")
    btn_reset = Button(ax_btn_reset, "Reset")

    state = {
        "ang_yz": 0.0,
        "ang_zx": 0.0,
        "rot_yx": float(default_rot_yx),
        "offset": 0.0,
    }

    def draw_once():
        inputs = DeformInput(state["ang_yz"], state["ang_zx"], state["rot_yx"])
        res = simulate_construct(inputs, L, L)

        # 3D view
        ax3d.cla()
        origin = np.zeros(3)
        # Colors
        col_Z_prox = 'tab:blue'
        col_Z_dist = 'tab:orange'
        col_A_prox = 'tab:green'
        col_A_dist = 'tab:red'
        axis_col = '0.6'  # gray for world axes
        # Unit directions for offset bases
        uZp = normalize(res.Z_prox_global)
        uZd = normalize(res.Z_dist_global)
        base_prox = origin + state["offset"] * uZp
        base_dist = origin + state["offset"] * uZd
        # Main axes
        ax3d.quiver(*origin, *res.Z_prox_global, length=1.0, normalize=False, arrow_length_ratio=0.1, color=col_Z_prox, linewidths=4)
        ax3d.text(*(res.Z_prox_global * 1.1), "Z_prox", color=col_Z_prox)
        ax3d.quiver(*origin, *res.Z_dist_global, length=1.0, normalize=False, arrow_length_ratio=0.1, color=col_Z_dist, linewidths=4)
        ax3d.text(*(res.Z_dist_global * 1.1), "Z_dist", color=col_Z_dist)
        # Anterior arrows from offset bases
        ax3d.quiver(*base_prox, *res.A_prox_global, length=1.0, normalize=False, arrow_length_ratio=0.1, color=col_A_prox, linewidths=4)
        ax3d.text(*(base_prox + res.A_prox_global * 1.1), "A_prox", color=col_A_prox)
        ax3d.quiver(*base_dist, *res.A_dist_global, length=1.0, normalize=False, arrow_length_ratio=0.1, color=col_A_dist, linewidths=4)
        ax3d.text(*(base_dist + res.A_dist_global * 1.1), "A_dist", color=col_A_dist)
        # World axes in neutral color
        axis_lim = 1.2
        ax3d.plot([0, axis_lim], [0, 0], [0, 0], color=axis_col); ax3d.text(axis_lim,0,0,'X', color=axis_col)
        ax3d.plot([0, 0], [0, axis_lim], [0, 0], color=axis_col); ax3d.text(0,axis_lim,0,'Y', color=axis_col)
        ax3d.plot([0, 0], [0, 0], [0, axis_lim], color=axis_col); ax3d.text(0,0,axis_lim,'Z', color=axis_col)
        ax3d.set_xlim(-axis_lim, axis_lim); ax3d.set_ylim(-axis_lim, axis_lim); ax3d.set_zlim(-axis_lim, axis_lim)
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        ax3d.set_title(f"3D (YZ={state['ang_yz']:.1f}°, ZX={state['ang_zx']:.1f}°, YX={state['rot_yx']:.1f}°, offset={state['offset']:.2f})")

        # XY projection
        axxy.cla()
        Aprox_xy = res.A_prox_global.copy(); Aprox_xy[2] = 0.0
        Adist_xy = res.A_dist_global.copy(); Adist_xy[2] = 0.0
        circle = plt.Circle((0,0), 1.0, fill=False); axxy.add_artist(circle)
        axxy.arrow(0, 0, Aprox_xy[0], Aprox_xy[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_A_prox)
        axxy.arrow(0, 0, Adist_xy[0], Adist_xy[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_A_dist)
        axxy.set_aspect('equal', adjustable='box')
        axxy.set_xlim(-1.2, 1.2); axxy.set_ylim(-1.2, 1.2)
        subtitle = f"Apparent axial angle = {res.apparent_xy_deg:.2f}°"
        axxy.set_title(f"XY Projection\n{subtitle}")
        axxy.set_xlabel("X"); axxy.set_ylabel("Y")

        # ZX projection (anterior + main axes), Z vertical
        axzx.cla()
        # anterior
        Aprox_zx = np.array([res.A_prox_global[0], res.A_prox_global[2]])  # (X, Z)
        Adist_zx = np.array([res.A_dist_global[0], res.A_dist_global[2]])
        axzx.arrow(0, 0, Aprox_zx[0], Aprox_zx[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_A_prox)
        axzx.arrow(0, 0, Adist_zx[0], Adist_zx[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_A_dist)
        # main axes
        Zprox_zx = np.array([res.Z_prox_global[0], res.Z_prox_global[2]])  # (X, Z)
        Zdist_zx = np.array([res.Z_dist_global[0], res.Z_dist_global[2]])
        axzx.arrow(0, 0, Zprox_zx[0], Zprox_zx[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_Z_prox)
        axzx.arrow(0, 0, Zdist_zx[0], Zdist_zx[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_Z_dist)
        axzx.set_aspect('equal', adjustable='box')
        axzx.set_xlim(-1.2, 1.2); axzx.set_ylim(-1.2, 1.2)
        axzx.set_xlabel('X'); axzx.set_ylabel('Z')
        axzx.set_title('ZX Projection (Z vertical)')

        # ZY projection (anterior + main axes), Z vertical
        axzy.cla()
        # anterior
        Aprox_zy = np.array([res.A_prox_global[1], res.A_prox_global[2]])  # (Y, Z)
        Adist_zy = np.array([res.A_dist_global[1], res.A_dist_global[2]])
        axzy.arrow(0, 0, Aprox_zy[0], Aprox_zy[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_A_prox)
        axzy.arrow(0, 0, Adist_zy[0], Adist_zy[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_A_dist)
        # main axes
        Zprox_zy = np.array([res.Z_prox_global[1], res.Z_prox_global[2]])  # (Y, Z)
        Zdist_zy = np.array([res.Z_dist_global[1], res.Z_dist_global[2]])
        axzy.arrow(0, 0, Zprox_zy[0], Zprox_zy[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_Z_prox)
        axzy.arrow(0, 0, Zdist_zy[0], Zdist_zy[1], width=0.04, head_width=0.12, length_includes_head=True, color=col_Z_dist)
        axzy.set_aspect('equal', adjustable='box')
        axzy.set_xlim(-1.2, 1.2); axzy.set_ylim(-1.2, 1.2)
        axzy.set_xlabel('Y'); axzy.set_ylabel('Z')
        axzy.set_title('ZY Projection (Z vertical)')

        fig.canvas.draw_idle()

    # Callbacks
    def on_yz(val):
        state["ang_yz"] = float(val); draw_once()
    def on_zx(val):
        state["ang_zx"] = float(val); draw_once()
    def on_off(val):
        state["offset"] = float(val); draw_once()
    def on_rot(text):
        try:
            state["rot_yx"] = float(text)
        except Exception:
            # keep previous if parse fails
            pass
        draw_once()
    def on_reset(event):
        s_yz.reset(); s_zx.reset(); s_off.reset();
        tb_rot.set_val(f"{default_rot_yx:.2f}")

    s_yz.on_changed(on_yz)
    s_zx.on_changed(on_zx)
    s_off.on_changed(on_off)
    tb_rot.on_submit(on_rot)
    btn_reset.on_clicked(on_reset)

    # Initial draw
    draw_once()
    plt.show()

# ---------------------------
# Runner
# ---------------------------

def run_example(ang_yz_deg, ang_zx_deg, rot_yx_deg, L=1.0):
    res = simulate_construct(DeformInput(ang_yz_deg, ang_zx_deg, rot_yx_deg), L, L)
    print(f"Inputs: YZ ang={ang_yz_deg}°, ZX ang={ang_zx_deg}°, YX rot={rot_yx_deg}°")
    print(f"Apparent axial rotation (XY) = {res.apparent_xy_deg:.4f}°")
    draw_3d_construct(res, title='3D Construct (two colinear Z segments joined at origin)')
    draw_xy_projection(res, title_prefix='XY Projection of anterior vectors')
    draw_zx_projection(res, title_prefix='ZX Projection of main axes')
    return res

def _prompt_float(msg, default=None):
    while True:
        try:
            s = input(f"{msg}" + (f" [{default}]" if default is not None else "") + ": ").strip()
            if s == "" and default is not None:
                return float(default)
            return float(s)
        except Exception:
            print("Please enter a numeric value.")

def main():
    parser = argparse.ArgumentParser(description="Apparent rotation visualizer for two connected segments.")
    parser.add_argument("--ang_yz", type=float, help="Angulation in YZ plane (deg), rotation about X.")
    parser.add_argument("--ang_zx", type=float, help="Angulation in ZX plane (deg), rotation about Y.")
    parser.add_argument("--rot_yx", type=float, help="Rotation in YX plane (deg), rotation about Z.")
    parser.add_argument("--length", type=float, default=1.0, help="Segment length (both segments).")
    parser.add_argument("--animate", action="store_true", help="Animate angulation progression (true axial rotation fixed).")
    parser.add_argument("--frames", type=int, default=60, help="Number of animation frames (when --animate is used).")
    parser.add_argument("--panel", action="store_true", help="Open interactive panel with sliders for angulation and a rotation input box.")
    args = parser.parse_args()

    ang_yz = args.ang_yz if args.ang_yz is not None else 0.0
    ang_zx = args.ang_zx if args.ang_zx is not None else 0.0
    rot_yx = args.rot_yx if args.rot_yx is not None else 0.0

    if args.panel or (args.ang_yz is None and args.ang_zx is None and args.rot_yx is None and not args.animate):
        run_panel(default_rot_yx=0.0, L=args.length)
    elif args.animate:
        run_animation(ang_yz, ang_zx, rot_yx, L=args.length, frames=args.frames)
    else:
        run_example(ang_yz, ang_zx, rot_yx, L=args.length)

if __name__ == "__main__":
    main()
