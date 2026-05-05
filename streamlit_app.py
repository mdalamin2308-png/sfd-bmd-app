import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — prevents ScriptRunContext warning
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch

st.set_page_config(page_title="SFD & BMD Solver", layout="wide", page_icon="📐")

st.markdown("""
<style>
@keyframes marquee {
    0%   { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}
.header-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 22px 32px 16px 32px;
    margin-bottom: 10px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
.logo-inline {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    justify-content: center;
    width: 100%;
}
.logo-svg {
    flex-shrink: 0;
}
.header-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #f7971e, #ffd200, #56ccf2, #2f80ed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.25;
}
.dev-line {
    text-align: center;
    font-size: 0.92rem;
    color: #b0c4de;
    margin: 6px 0 0 0;
}
.dev-line strong {
    color: #ffd200;
}
.marquee-wrap {
    overflow: hidden;
    white-space: nowrap;
    margin-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 7px;
}
.marquee-text {
    display: inline-block;
    animation: marquee 22s linear infinite;
    font-size: 0.82rem;
    color: #90b8d0;
}
.marquee-text strong { color: #56ccf2; }
</style>

<div class="header-box">
  <div class="logo-inline">
    <!-- SVG logo: beam with SFD & BMD curves -->
    <svg class="logo-svg" width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Beam -->
      <rect x="4" y="33" width="64" height="6" rx="2" fill="#56ccf2" opacity="0.85"/>
      <!-- Left pin support -->
      <polygon points="12,39 6,50 18,50" fill="#f7971e" opacity="0.9"/>
      <!-- Right roller support -->
      <polygon points="60,39 54,50 66,50" fill="#f7971e" opacity="0.9"/>
      <ellipse cx="60" cy="52" rx="4" ry="2.5" fill="#ffd200" opacity="0.8"/>
      <!-- Point load arrow -->
      <line x1="36" y1="10" x2="36" y2="32" stroke="#ff6b6b" stroke-width="2.5" stroke-linecap="round"/>
      <polygon points="36,33 33,24 39,24" fill="#ff6b6b"/>
      <!-- SFD curve (below beam) -->
      <polyline points="4,58 22,58 22,66 50,66 50,58 68,58" stroke="#ffd200" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      <!-- BMD curve hint -->
      <path d="M4 58 Q36 44 68 58" stroke="#2f80ed" stroke-width="1.5" fill="none" stroke-dasharray="3,2" opacity="0.7"/>
    </svg>
    <h1 class="header-title">Shear Force &amp; Bending Moment<br>Diagram Calculation Application</h1>
    <svg class="logo-svg" width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Mirror of left logo for symmetry -->
      <rect x="4" y="33" width="64" height="6" rx="2" fill="#56ccf2" opacity="0.85"/>
      <polygon points="12,39 6,50 18,50" fill="#f7971e" opacity="0.9"/>
      <polygon points="60,39 54,50 66,50" fill="#f7971e" opacity="0.9"/>
      <ellipse cx="60" cy="52" rx="4" ry="2.5" fill="#ffd200" opacity="0.8"/>
      <!-- UDL arrows -->
      <line x1="20" y1="12" x2="20" y2="32" stroke="#56ccf2" stroke-width="2" stroke-linecap="round"/>
      <line x1="30" y1="10" x2="30" y2="32" stroke="#56ccf2" stroke-width="2" stroke-linecap="round"/>
      <line x1="40" y1="10" x2="40" y2="32" stroke="#56ccf2" stroke-width="2" stroke-linecap="round"/>
      <line x1="50" y1="12" x2="50" y2="32" stroke="#56ccf2" stroke-width="2" stroke-linecap="round"/>
      <line x1="18" y1="10" x2="52" y2="10" stroke="#56ccf2" stroke-width="2" stroke-linecap="round"/>
      <!-- BMD parabola -->
      <path d="M4 58 Q36 70 68 58" stroke="#56ccf2" stroke-width="2" fill="rgba(86,204,242,0.12)" stroke-linecap="round"/>
    </svg>
  </div>
  <p class="dev-line">Developed by <strong>Md. Al Amin</strong></p>
  <div class="marquee-wrap">
    <span class="marquee-text">
      &#x2605;&nbsp; This application is dedicated to all students of the Department of Civil Engineering at&nbsp;
      <strong>Bangladesh Army University of Engineering and Technology (BAUET)</strong>
      &nbsp;&#x2605;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
      &#x2605;&nbsp; This application is dedicated to all students of the Department of Civil Engineering at&nbsp;
      <strong>Bangladesh Army University of Engineering and Technology (BAUET)</strong>
      &nbsp;&#x2605;
    </span>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Session state init ────────────────────────
if "point_loads" not in st.session_state:
    st.session_state.point_loads = [{"P": 20.0, "x": 4.0}]
if "udls" not in st.session_state:
    st.session_state.udls = [{"w": 5.0, "xs": 6.0, "xe": 10.0}]
if "uvls" not in st.session_state:
    st.session_state.uvls = []
if "moments" not in st.session_state:
    st.session_state.moments = []
if "beam_length_m" not in st.session_state:
    st.session_state.beam_length_m = 10.0
if "overhang_m" not in st.session_state:
    st.session_state.overhang_m = 2.0
if "length_unit" not in st.session_state:
    st.session_state.length_unit = "m"
if "force_unit" not in st.session_state:
    st.session_state.force_unit = "kN"
if "support_a_m" not in st.session_state:
    st.session_state.support_a_m = 0.0
if "support_b_m" not in st.session_state:
    st.session_state.support_b_m = 10.0
if "prev_length_unit" not in st.session_state:
    st.session_state.prev_length_unit = st.session_state.length_unit
if "prev_force_unit" not in st.session_state:
    st.session_state.prev_force_unit = st.session_state.force_unit
if "units_changed" not in st.session_state:
    st.session_state.units_changed = False

LENGTH_FACTORS_M = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "in": 0.0254,
}

FORCE_FACTORS_KN = {
    "kN": 1.0,
    "N": 0.001,
    "MN": 1000.0,
    "lb": 0.0044482216,
    "kip": 4.4482216,
}

# ═══════════════════════════════════════════════
#  LIVE BEAM DRAWING FUNCTION
# ═══════════════════════════════════════════════
def draw_beam_figure(L, beam_type, support_a, support_b,
                     point_loads, udls, uvls, moments,
                     length_unit, force_unit, length_factor_m, force_factor_kn):
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_xlim(-0.8, L + 0.8)
    ax.set_ylim(-2.2, 3.2)
    ax.set_aspect("auto")
    ax.axis("off")

    # ── Ground hatching ───────────────────────
    def hatch_ground(x_center, y_base, width=0.7):
        ax.plot([x_center - width, x_center + width],
                [y_base, y_base], color="#888", lw=1.5)
        for xi in np.linspace(x_center - width, x_center + width - 0.15, 7):
            ax.plot([xi, xi + 0.15], [y_base, y_base - 0.18], color="#888", lw=0.8)

    # ── Support drawing helpers ────────────────
    def draw_pin(x, name=""):
        tri = plt.Polygon([[x, -0.18], [x - 0.35, -0.72], [x + 0.35, -0.72]],
                          closed=True, facecolor="#f0a500", edgecolor="white", lw=1)
        ax.add_patch(tri)
        hatch_ground(x, -0.72)
        lbl = f"{name} (Pin)\nx={x / length_factor_m:.2f}{length_unit}" if name else f"Pin\nx={x / length_factor_m:.2f}{length_unit}"
        ax.text(x, -1.05, lbl, ha="center", va="top", color="#f0a500", fontsize=7)
        if name:
            ax.text(x, 0.28, name, ha="center", va="bottom", color="#f0a500",
                    fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", facecolor="#1a2a0a",
                              edgecolor="#f0a500", lw=1.5))

    def draw_roller(x, name=""):
        tri = plt.Polygon([[x, -0.18], [x - 0.3, -0.55], [x + 0.3, -0.55]],
                          closed=True, facecolor="#50c878", edgecolor="white", lw=1)
        ax.add_patch(tri)
        circ = plt.Circle((x, -0.75), 0.18, facecolor="#50c878",
                          edgecolor="white", lw=1)
        ax.add_patch(circ)
        hatch_ground(x, -0.95)
        lbl = f"{name} (Roller)\nx={x / length_factor_m:.2f}{length_unit}" if name else f"Roller\nx={x / length_factor_m:.2f}{length_unit}"
        ax.text(x, -1.2, lbl, ha="center", va="top", color="#50c878", fontsize=7)
        if name:
            ax.text(x, 0.28, name, ha="center", va="bottom", color="#50c878",
                    fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", facecolor="#0a1a10",
                              edgecolor="#50c878", lw=1.5))

    def draw_fixed(x, side="left", name=""):
        d = -0.25 if side == "left" else 0.25
        ax.plot([x, x], [-1.0, 1.0], color="#f77f00", linewidth=7, solid_capstyle="butt")
        for dy in np.linspace(-0.8, 0.8, 7):
            ax.plot([x, x + d], [dy, dy - 0.18], color="#f77f00", lw=1)
        lbl = f"{name} (Fixed)\nx={x / length_factor_m:.2f}{length_unit}" if name else f"Fixed\nx={x / length_factor_m:.2f}{length_unit}"
        ax.text(x + d * 2.5, -0.55, lbl, ha="center", va="center", color="#f77f00", fontsize=7)
        if name:
            ax.text(x + d * 1.8, 0.55, name, ha="center", va="bottom", color="#f77f00",
                    fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", facecolor="#1a0e00",
                              edgecolor="#f77f00", lw=1.5))

    def draw_free(x, name=""):
        ax.plot(x, 0, "o", color="#aaa", markersize=6)
        lbl = f"{name} (Free)\nx={x / length_factor_m:.2f}{length_unit}" if name else f"Free\nx={x / length_factor_m:.2f}{length_unit}"
        ax.text(x + 0.15, -0.4, lbl, ha="left", color="#aaa", fontsize=7)
        if name:
            ax.text(x, 0.28, name, ha="center", va="bottom", color="#aaa",
                    fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", facecolor="#1a1a1a",
                              edgecolor="#aaa", lw=1.5))

    # ── Beam body ─────────────────────────────
    beam = mpatches.FancyBboxPatch(
        (0, -0.18), L, 0.36,
        boxstyle="round,pad=0.04",
        linewidth=1.5, edgecolor="#aaa", facecolor="#2a4a6a")
    ax.add_patch(beam)

    # dimension line
    ax.annotate("", xy=(L, -1.55), xytext=(0, -1.55),
                arrowprops=dict(arrowstyle="<->", color="#aaa", lw=1))
    ax.text(L / 2, -1.7, f"L = {L / length_factor_m:.2f} {length_unit}", ha="center", color="#aaa", fontsize=8)

    # x-axis ticks (show in user-selected length unit)
    L_disp_beam = L / length_factor_m
    n_ticks = max(5, min(20, int(L_disp_beam) + 1))
    for xi in np.linspace(0, L, n_ticks):
        ax.plot([xi, xi], [-0.18, -0.28], color="#666", lw=0.8)
        ax.text(xi, -0.38, f"{xi/length_factor_m:.3g}", ha="center", va="top", color="#666", fontsize=6.5)

    # ── Draw supports ─────────────────────────
    if beam_type == "Simply Supported":
        draw_pin(support_a, "A")
        draw_roller(support_b, "B")
    elif beam_type == "Cantilever (Fixed-Free)":
        draw_fixed(0, "left", "A")
        draw_free(L, "B")
    elif beam_type == "Propped Cantilever":
        draw_fixed(support_a, "left", "A")
        draw_roller(support_b, "B")
    elif beam_type == "Fixed-Fixed":
        draw_fixed(support_a, "left", "A")
        draw_fixed(support_b, "right", "B")
    elif beam_type == "Overhanging (Left)":
        draw_pin(support_a, "A")
        draw_roller(support_b, "B")
    elif beam_type == "Overhanging (Right)":
        draw_pin(support_a, "A")
        draw_roller(support_b, "B")

    # ── Draw UDLs ─────────────────────────────
    for udl in udls:
        w_val, w_s, w_e = udl["w"], udl["xs"], udl["xe"]
        if w_e <= w_s:
            continue
        h_arr = 1.1
        positions = np.linspace(w_s, w_e, max(int((w_e - w_s) * 3) + 2, 5))
        for xi in positions:
            ax.annotate("", xy=(xi, 0.18), xytext=(xi, 0.18 + h_arr),
                        arrowprops=dict(arrowstyle="-|>", color="#00bfff",
                                        lw=1.3, mutation_scale=8))
        ax.plot([w_s, w_e], [0.18 + h_arr, 0.18 + h_arr], color="#00bfff", lw=2)
        ax.plot([w_s, w_s], [0.18, 0.18 + h_arr], color="#00bfff", lw=1, ls="--")
        ax.plot([w_e, w_e], [0.18, 0.18 + h_arr], color="#00bfff", lw=1, ls="--")
        ax.text((w_s + w_e) / 2, 0.18 + h_arr + 0.12,
                f"UDL {w_val / (force_factor_kn / length_factor_m):.2f} {force_unit}/{length_unit}\n"
                f"[{w_s / length_factor_m:.2f}–{w_e / length_factor_m:.2f} {length_unit}]",
                ha="center", va="bottom", color="#00bfff", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0e1117",
                          edgecolor="#00bfff", alpha=0.8))

    # ── Draw UVLs ─────────────────────────────
    for uvl in uvls:
        w0, w1, uvl_s, uvl_e = uvl["w0"], uvl["w1"], uvl["xs"], uvl["xe"]
        if uvl_e <= uvl_s:
            continue
        positions = np.linspace(uvl_s, uvl_e, max(int((uvl_e - uvl_s) * 3) + 2, 6))
        for xi in positions:
            t = (xi - uvl_s) / (uvl_e - uvl_s)
            h_arr = 0.25 + (w0 + (w1 - w0) * t) * 0.06
            ax.annotate("", xy=(xi, 0.18), xytext=(xi, 0.18 + h_arr),
                        arrowprops=dict(arrowstyle="-|>", color="#ff6eb4",
                                        lw=1.0, mutation_scale=7))
        heights = [0.25 + (w0 + (w1 - w0) * (xi - uvl_s) / (uvl_e - uvl_s)) * 0.06
                   for xi in positions]
        ax.plot(positions, np.array(heights) + 0.18, color="#ff6eb4", lw=1.5)
        ax.text((uvl_s + uvl_e) / 2, max(heights) + 0.18 + 0.12,
                f"UVL {w0 / (force_factor_kn / length_factor_m):.2f}→{w1 / (force_factor_kn / length_factor_m):.2f} {force_unit}/{length_unit}\n"
                f"[{uvl_s / length_factor_m:.2f}–{uvl_e / length_factor_m:.2f} {length_unit}]",
                ha="center", va="bottom", color="#ff6eb4", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0e1117",
                          edgecolor="#ff6eb4", alpha=0.8))

    # ── Draw Point Loads ──────────────────────
    for pl in point_loads:
        p_val, p_pos = pl["P"], pl["x"]
        ax.annotate("", xy=(p_pos, 0.18), xytext=(p_pos, 1.8),
                    arrowprops=dict(arrowstyle="-|>", color="#ff4444",
                                    lw=2.5, mutation_scale=12))
        ax.text(p_pos, 2.0, f"{p_val / force_factor_kn:.2f} {force_unit}\nx={p_pos / length_factor_m:.2f}{length_unit}",
                ha="center", va="bottom", color="#ff4444", fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0e1117",
                          edgecolor="#ff4444", alpha=0.85))

    # ── Draw Point Moments ────────────────────
    for mom in moments:
        m_val, m_pos = mom["M"], mom["x"]
        arc = Arc((m_pos, 0.3), 0.55, 0.55, angle=0,
                  theta1=30, theta2=330, color="#ffd700", lw=2)
        ax.add_patch(arc)
        ax.annotate("", xy=(m_pos + 0.27, 0.56),
                    xytext=(m_pos + 0.27, 0.58),
                    arrowprops=dict(arrowstyle="-|>", color="#ffd700",
                                    lw=1.5, mutation_scale=10))
        ax.text(m_pos, 1.1, f"M={m_val / (force_factor_kn * length_factor_m):.2f} {force_unit}\N{MIDDLE DOT}{length_unit}\nx={m_pos / length_factor_m:.2f}{length_unit}",
                ha="center", color="#ffd700", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0e1117",
                          edgecolor="#ffd700", alpha=0.85))

    # ── Beam label ────────────────────────────
    ax.text(L / 2, -0.0, "BEAM", ha="center", va="center",
            color="#8ab0cc", fontsize=9, alpha=0.5)

    plt.tight_layout(pad=0)
    return fig


# ═══════════════════════════════════════════════
#  INPUT PANEL — LEFT (beam setup) + RIGHT (loads)
# ═══════════════════════════════════════════════
col_setup, col_loads = st.columns([1, 2])

with col_setup:
    st.markdown("### Beam Setup")

    length_unit = st.selectbox("Length Unit", list(LENGTH_FACTORS_M.keys()),
                               index=list(LENGTH_FACTORS_M.keys()).index(st.session_state.length_unit))
    force_unit = st.selectbox("Load Unit", list(FORCE_FACTORS_KN.keys()),
                              index=list(FORCE_FACTORS_KN.keys()).index(st.session_state.force_unit))

    st.session_state.units_changed = (
        length_unit != st.session_state.prev_length_unit
        or force_unit != st.session_state.prev_force_unit
    )
    st.session_state.length_unit = length_unit
    st.session_state.force_unit = force_unit
    st.session_state.prev_length_unit = length_unit
    st.session_state.prev_force_unit = force_unit

    length_factor_m = LENGTH_FACTORS_M[length_unit]
    force_factor_kn = FORCE_FACTORS_KN[force_unit]

    L_input = st.number_input(
        f"Beam Length ({length_unit})",
        value=float(st.session_state.beam_length_m / length_factor_m),
        min_value=float(0.5 / length_factor_m),
        step=float(max(0.1, 0.5 / length_factor_m)),
    )
    L = float(L_input) * length_factor_m
    st.session_state.beam_length_m = L

    beam_type = st.selectbox("Support Type", [
        "Simply Supported",
        "Cantilever (Fixed-Free)",
        "Propped Cantilever",
        "Fixed-Fixed",
        "Overhanging (Left)",
        "Overhanging (Right)",
    ])

    # Support icon legend
    support_icons = {
        "Simply Supported":        "📌 Pin (A)  +  🟢 Roller (B)  — set positions below",
        "Cantilever (Fixed-Free)": "🟧 Fixed at x=0  +  ○ Free at x=L",
        "Propped Cantilever":      "🟧 Fixed (A)  +  🟢 Roller (B)  — set positions below",
        "Fixed-Fixed":             "🟧 Fixed (A)  +  🟧 Fixed (B)  — set positions below",
        "Overhanging (Left)":      "📌 Pin (A, inset from left end)  +  🟢 Roller at x=L",
        "Overhanging (Right)":     "📌 Pin at x=0  +  🟢 Roller (B, inset from right end)",
    }
    st.caption(support_icons[beam_type])

    if beam_type == "Cantilever (Fixed-Free)":
        support_a = 0.0
        support_b = L
    elif beam_type == "Overhanging (Left)":
        a_default = min(max(float(st.session_state.support_a_m), 0.0), float(L - 0.01))
        a_input = st.number_input(
            f"Support A location ({length_unit})  [0 – {round((L-0.01)/length_factor_m, 4)}]",
            value=float(a_default / length_factor_m),
            min_value=0.0,
            step=0.01,
        )
        support_a = min(max(float(a_input) * length_factor_m, 0.0), L - 0.01)
        support_b = L
    elif beam_type == "Overhanging (Right)":
        b_default = min(max(float(st.session_state.support_b_m), 0.01), float(L - 0.01))
        b_input = st.number_input(
            f"Support B location ({length_unit})  [0.01 – {round((L-0.01)/length_factor_m, 4)}]",
            value=float(b_default / length_factor_m),
            min_value=0.0,
            step=0.01,
        )
        support_a = 0.0
        support_b = min(max(float(b_input) * length_factor_m, 0.01), L - 0.01)
    else:
        a_default = min(max(float(st.session_state.support_a_m), 0.0), float(L - 0.01))
        a_input = st.number_input(
            f"Support A location ({length_unit})  [0 – {round(L/length_factor_m, 4)}]",
            value=float(a_default / length_factor_m),
            min_value=0.0,
            step=0.01,
        )
        support_a = min(max(float(a_input) * length_factor_m, 0.0), L - 0.01)

        b_default = min(max(float(st.session_state.support_b_m), 0.0), float(L))
        b_input = st.number_input(
            f"Support B location ({length_unit})  [0 – {round(L/length_factor_m, 4)}]",
            value=float(b_default / length_factor_m),
            min_value=0.0,
            step=0.01,
        )
        support_b = min(max(float(b_input) * length_factor_m, support_a + 0.01), L)

    st.session_state.support_a_m = support_a
    st.session_state.support_b_m = support_b
    overhang = max(0.0, support_a) if beam_type == "Overhanging (Left)" else max(0.0, L - support_b)

# ── Pre-sync load values from widget keys so diagram is up-to-date ──
for _i, _pl in enumerate(st.session_state.point_loads):
    if f"p_{_i}" in st.session_state:
        _pl["P"] = float(st.session_state[f"p_{_i}"]) * force_factor_kn
    if f"px_{_i}" in st.session_state:
        _pl["x"] = float(st.session_state[f"px_{_i}"]) * length_factor_m
for _i, _u in enumerate(st.session_state.udls):
    if f"w_{_i}" in st.session_state:
        _u["w"] = float(st.session_state[f"w_{_i}"]) * (force_factor_kn / length_factor_m)
    if f"ws_{_i}" in st.session_state:
        _u["xs"] = float(st.session_state[f"ws_{_i}"]) * length_factor_m
    if f"we_{_i}" in st.session_state:
        _u["xe"] = float(st.session_state[f"we_{_i}"]) * length_factor_m
for _i, _v in enumerate(st.session_state.uvls):
    if f"uvl_w0_{_i}" in st.session_state:
        _v["w0"] = float(st.session_state[f"uvl_w0_{_i}"]) * (force_factor_kn / length_factor_m)
    if f"uvl_w1_{_i}" in st.session_state:
        _v["w1"] = float(st.session_state[f"uvl_w1_{_i}"]) * (force_factor_kn / length_factor_m)
    if f"uvl_s_{_i}" in st.session_state:
        _v["xs"] = float(st.session_state[f"uvl_s_{_i}"]) * length_factor_m
    if f"uvl_e_{_i}" in st.session_state:
        _v["xe"] = float(st.session_state[f"uvl_e_{_i}"]) * length_factor_m
for _i, _m in enumerate(st.session_state.moments):
    if f"m_{_i}" in st.session_state:
        _m["M"] = float(st.session_state[f"m_{_i}"]) * (force_factor_kn * length_factor_m)
    if f"mx_{_i}" in st.session_state:
        _m["x"] = float(st.session_state[f"mx_{_i}"]) * length_factor_m

# ═══════════════════════════════════════════════
#  LIVE BEAM DIAGRAM
# ═══════════════════════════════════════════════
st.markdown("### Live Beam Diagram")
fig_beam = draw_beam_figure(
    L, beam_type, support_a, support_b,
    st.session_state.point_loads,
    st.session_state.udls,
    st.session_state.uvls,
    st.session_state.moments,
    length_unit, force_unit, length_factor_m, force_factor_kn
)
st.pyplot(fig_beam, width='stretch')
plt.close(fig_beam)

with col_loads:
    st.markdown("### Loads")
    L_disp = float(L / length_factor_m)

    def clear_keys(prefixes):
        keys_to_del = [k for k in st.session_state.keys() if any(k.startswith(p) for p in prefixes)]
        for k in keys_to_del:
            del st.session_state[k]

    if st.session_state.units_changed:
        clear_keys([
            "p_", "px_",
            "w_", "ws_", "we_",
            "uvl_w0_", "uvl_w1_", "uvl_s_", "uvl_e_",
            "m_", "mx_",
        ])
        st.session_state.units_changed = False

    # If beam length is reduced, keep previous widget values within valid bounds.
    for i, pl in enumerate(st.session_state.point_loads):
        pl["x"] = min(float(pl["x"]), float(L))
        px_key = f"px_{i}"
        if px_key in st.session_state and float(st.session_state[px_key]) > L_disp:
            del st.session_state[px_key]
    for i, udl in enumerate(st.session_state.udls):
        udl["xs"] = min(float(udl["xs"]), float(L))
        udl["xe"] = min(float(udl["xe"]), float(L))
        ws_key = f"ws_{i}"
        we_key = f"we_{i}"
        if ws_key in st.session_state and float(st.session_state[ws_key]) > L_disp:
            del st.session_state[ws_key]
        if we_key in st.session_state and float(st.session_state[we_key]) > L_disp:
            del st.session_state[we_key]
    for i, uvl in enumerate(st.session_state.uvls):
        uvl["xs"] = min(float(uvl["xs"]), float(L))
        uvl["xe"] = min(float(uvl["xe"]), float(L))
        uvl_s_key = f"uvl_s_{i}"
        uvl_e_key = f"uvl_e_{i}"
        if uvl_s_key in st.session_state and float(st.session_state[uvl_s_key]) > L_disp:
            del st.session_state[uvl_s_key]
        if uvl_e_key in st.session_state and float(st.session_state[uvl_e_key]) > L_disp:
            del st.session_state[uvl_e_key]
    for i, mom in enumerate(st.session_state.moments):
        mom["x"] = min(float(mom["x"]), float(L))
        mx_key = f"mx_{i}"
        if mx_key in st.session_state and float(st.session_state[mx_key]) > L_disp:
            del st.session_state[mx_key]

    tab_pl, tab_udl, tab_uvl, tab_mom = st.tabs(
        ["Point Loads", "UDL", "UVL (Varying)", "Moments"])

    # ── Point Loads ───────────────────────────
    with tab_pl:
        st.caption("Add/remove point loads. Downward = positive.")
        for i, pl in enumerate(st.session_state.point_loads):
            c1, c2, c3 = st.columns([2, 2, 1])
            p_key = f"p_{i}"
            px_key = f"px_{i}"
            if p_key not in st.session_state:
                st.session_state[p_key] = float(pl["P"] / force_factor_kn)
            if px_key not in st.session_state:
                st.session_state[px_key] = min(float(pl["x"] / length_factor_m), L_disp)
            st.session_state.point_loads[i]["P"] = c1.number_input(
                f"P{i+1} ({force_unit})", key=p_key) * force_factor_kn
            st.session_state.point_loads[i]["x"] = c2.number_input(
                f"x{i+1} ({length_unit})", min_value=0.0, max_value=L_disp, key=px_key) * length_factor_m
            if c3.button("✕", key=f"del_p_{i}", help="Remove"):
                st.session_state.point_loads.pop(i)
                clear_keys(["p_", "px_"])
                st.rerun()
        if st.button("+ Add Point Load"):
            st.session_state.point_loads.append({"P": 10.0, "x": L / 2})
            clear_keys(["p_", "px_"])
            st.rerun()

    # ── UDLs ──────────────────────────────────
    with tab_udl:
        st.caption("Uniformly distributed loads. Downward = positive.")
        for i, udl in enumerate(st.session_state.udls):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            w_key = f"w_{i}"
            ws_key = f"ws_{i}"
            we_key = f"we_{i}"
            if w_key not in st.session_state:
                st.session_state[w_key] = float(udl["w"] / (force_factor_kn / length_factor_m))
            if ws_key not in st.session_state:
                st.session_state[ws_key] = min(float(udl["xs"] / length_factor_m), L_disp)
            if we_key not in st.session_state:
                st.session_state[we_key] = min(float(udl["xe"] / length_factor_m), L_disp)
            st.session_state.udls[i]["w"] = c1.number_input(
                f"w{i+1} ({force_unit}/{length_unit})", key=w_key) * (force_factor_kn / length_factor_m)
            st.session_state.udls[i]["xs"] = c2.number_input(
                f"Start{i+1} ({length_unit})", min_value=0.0, max_value=L_disp, key=ws_key) * length_factor_m
            st.session_state.udls[i]["xe"] = c3.number_input(
                f"End{i+1} ({length_unit})", min_value=0.0, max_value=L_disp, key=we_key) * length_factor_m
            if c4.button("✕", key=f"del_u_{i}", help="Remove"):
                st.session_state.udls.pop(i)
                clear_keys(["w_", "ws_", "we_"])
                st.rerun()
        if st.button("+ Add UDL"):
            st.session_state.udls.append({"w": 5.0, "xs": 0.0, "xe": L})
            clear_keys(["w_", "ws_", "we_"])
            st.rerun()

    # ── UVLs ──────────────────────────────────
    with tab_uvl:
        st.caption("Triangular / trapezoidal loads (varying intensity).")
        for i, uvl in enumerate(st.session_state.uvls):
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
            w0_key = f"uvl_w0_{i}"
            w1_key = f"uvl_w1_{i}"
            uvl_s_key = f"uvl_s_{i}"
            uvl_e_key = f"uvl_e_{i}"
            if w0_key not in st.session_state:
                st.session_state[w0_key] = float(uvl["w0"] / (force_factor_kn / length_factor_m))
            if w1_key not in st.session_state:
                st.session_state[w1_key] = float(uvl["w1"] / (force_factor_kn / length_factor_m))
            if uvl_s_key not in st.session_state:
                st.session_state[uvl_s_key] = min(float(uvl["xs"] / length_factor_m), L_disp)
            if uvl_e_key not in st.session_state:
                st.session_state[uvl_e_key] = min(float(uvl["xe"] / length_factor_m), L_disp)
            st.session_state.uvls[i]["w0"] = c1.number_input(
                f"w_start{i+1} ({force_unit}/{length_unit})", key=w0_key) * (force_factor_kn / length_factor_m)
            st.session_state.uvls[i]["w1"] = c2.number_input(
                f"w_end{i+1} ({force_unit}/{length_unit})", key=w1_key) * (force_factor_kn / length_factor_m)
            st.session_state.uvls[i]["xs"] = c3.number_input(
                f"Start{i+1} ({length_unit})", min_value=0.0, max_value=L_disp, key=uvl_s_key) * length_factor_m
            st.session_state.uvls[i]["xe"] = c4.number_input(
                f"End{i+1} ({length_unit})", min_value=0.0, max_value=L_disp, key=uvl_e_key) * length_factor_m
            if c5.button("✕", key=f"del_uvl_{i}", help="Remove"):
                st.session_state.uvls.pop(i)
                clear_keys(["uvl_w0_", "uvl_w1_", "uvl_s_", "uvl_e_"])
                st.rerun()
        if st.button("+ Add UVL"):
            st.session_state.uvls.append({"w0": 0.0, "w1": 10.0, "xs": 0.0, "xe": L})
            clear_keys(["uvl_w0_", "uvl_w1_", "uvl_s_", "uvl_e_"])
            st.rerun()

    # ── Point Moments ─────────────────────────
    with tab_mom:
        st.caption("Point moments. Clockwise = positive.")
        for i, mom in enumerate(st.session_state.moments):
            c1, c2, c3 = st.columns([2, 2, 1])
            m_key = f"m_{i}"
            mx_key = f"mx_{i}"
            if m_key not in st.session_state:
                st.session_state[m_key] = float(mom["M"] / (force_factor_kn * length_factor_m))
            if mx_key not in st.session_state:
                st.session_state[mx_key] = min(float(mom["x"] / length_factor_m), L_disp)
            st.session_state.moments[i]["M"] = c1.number_input(
                f"M{i+1} ({force_unit}·{length_unit})", key=m_key) * (force_factor_kn * length_factor_m)
            st.session_state.moments[i]["x"] = c2.number_input(
                f"x{i+1} ({length_unit})", min_value=0.0, max_value=L_disp, key=mx_key) * length_factor_m
            if c3.button("✕", key=f"del_m_{i}", help="Remove"):
                st.session_state.moments.pop(i)
                clear_keys(["m_", "mx_"])
                st.rerun()
        if st.button("+ Add Moment"):
            st.session_state.moments.append({"M": 10.0, "x": L / 2})
            clear_keys(["m_", "mx_"])
            st.rerun()

# ═══════════════════════════════════════════════
#  CALCULATE
# ═══════════════════════════════════════════════
st.divider()
if st.button("Calculate SFD & BMD", type="primary"):

    point_loads = st.session_state.point_loads
    udls        = st.session_state.udls
    uvls        = st.session_state.uvls
    moments     = st.session_state.moments

    N = 2000
    x = np.linspace(0, L, N)

    # ── Total load & moment about A ────────────
    total_F   = sum(pl["P"] for pl in point_loads)
    total_M_A = sum(pl["P"] * pl["x"] for pl in point_loads)

    for u in udls:
        F = u["w"] * (u["xe"] - u["xs"])
        total_F   += F
        total_M_A += F * (u["xs"] + u["xe"]) / 2

    for v in uvls:
        length = v["xe"] - v["xs"]
        if length > 0:
            F = (v["w0"] + v["w1"]) / 2 * length
            denom = v["w0"] + v["w1"]
            cx = (v["xs"] + length * (v["w0"] + 2 * v["w1"]) / (3 * denom)
                  if abs(denom) > 1e-9 else v["xs"] + length / 2)
            total_F   += F
            total_M_A += F * cx

    for m in moments:
        total_M_A += m["M"]

    # ── Reactions ─────────────────────────────
    RA = RB = MA_fix = 0.0

    # Support positions
    xA = support_a
    xB_pos = support_b

    if beam_type == "Simply Supported":
        span = xB_pos - xA
        total_M_xA = sum(pl["P"] * (pl["x"] - xA) for pl in point_loads)
        for u in udls:
            F = u["w"] * (u["xe"] - u["xs"])
            total_M_xA += F * (((u["xs"] + u["xe"]) / 2) - xA)
        for v in uvls:
            length = v["xe"] - v["xs"]
            if length > 0:
                F = (v["w0"] + v["w1"]) / 2 * length
                denom = v["w0"] + v["w1"]
                cx = (v["xs"] + length * (v["w0"] + 2 * v["w1"]) / (3 * denom)
                      if abs(denom) > 1e-9 else v["xs"] + length / 2)
                total_M_xA += F * (cx - xA)
        total_M_xA += sum(m["M"] for m in moments)
        RB = total_M_xA / span
        RA = total_F - RB

    elif beam_type == "Cantilever (Fixed-Free)":
        RA     = total_F
        MA_fix = total_M_A

    elif beam_type == "Overhanging (Left)":
        span = xB_pos - xA
        total_M_xA = sum(pl["P"] * (pl["x"] - xA) for pl in point_loads)
        for u in udls:
            F = u["w"] * (u["xe"] - u["xs"])
            total_M_xA += F * (((u["xs"] + u["xe"]) / 2) - xA)
        for v in uvls:
            length = v["xe"] - v["xs"]
            if length > 0:
                F = (v["w0"] + v["w1"]) / 2 * length
                denom = v["w0"] + v["w1"]
                cx = (v["xs"] + length * (v["w0"] + 2 * v["w1"]) / (3 * denom)
                      if abs(denom) > 1e-9 else v["xs"] + length / 2)
                total_M_xA += F * (cx - xA)
        total_M_xA += sum(m["M"] for m in moments)
        RB = total_M_xA / span
        RA = total_F - RB

    elif beam_type == "Overhanging (Right)":
        span = xB_pos - xA
        total_M_xA = sum(pl["P"] * (pl["x"] - xA) for pl in point_loads)
        for u in udls:
            F = u["w"] * (u["xe"] - u["xs"])
            total_M_xA += F * (((u["xs"] + u["xe"]) / 2) - xA)
        for v in uvls:
            length = v["xe"] - v["xs"]
            if length > 0:
                F = (v["w0"] + v["w1"]) / 2 * length
                denom = v["w0"] + v["w1"]
                cx = (v["xs"] + length * (v["w0"] + 2 * v["w1"]) / (3 * denom)
                      if abs(denom) > 1e-9 else v["xs"] + length / 2)
                total_M_xA += F * (cx - xA)
        total_M_xA += sum(m["M"] for m in moments)
        RB = total_M_xA / span
        RA = total_F - RB

    elif beam_type in ["Propped Cantilever", "Fixed-Fixed"]:
        span = xB_pos - xA
        total_M_xA = sum(pl["P"] * (pl["x"] - xA) for pl in point_loads)
        for u in udls:
            F = u["w"] * (u["xe"] - u["xs"])
            total_M_xA += F * (((u["xs"] + u["xe"]) / 2) - xA)
        for v in uvls:
            length = v["xe"] - v["xs"]
            if length > 0:
                F = (v["w0"] + v["w1"]) / 2 * length
                denom = v["w0"] + v["w1"]
                cx = (v["xs"] + length * (v["w0"] + 2 * v["w1"]) / (3 * denom)
                      if abs(denom) > 1e-9 else v["xs"] + length / 2)
                total_M_xA += F * (cx - xA)
        total_M_xA += sum(m["M"] for m in moments)
        RB = total_M_xA / span
        RA = total_F - RB
        st.warning("Propped Cantilever / Fixed-Fixed are statically indeterminate. "
                   "Reactions are approximate.")

    # ── SFD & BMD arrays ──────────────────────
    V     = np.zeros(N)
    M_arr = np.zeros(N)

    for j, xi in enumerate(x):
        if beam_type == "Cantilever (Fixed-Free)":
            # Scan from free end (right) toward fixed end (left).
            # Downward loads to the right of xi create clockwise moment on
            # the right portion → hogging → NEGATIVE by standard convention.
            shear  = 0.0
            moment = 0.0
            for pl in point_loads:
                if xi <= pl["x"]:
                    shear  += pl["P"]
                    moment -= pl["P"] * (pl["x"] - xi)   # hogging → negative
            for u in udls:
                if xi < u["xe"]:
                    eff_s = max(xi, u["xs"])
                    if u["xe"] > eff_s:
                        F      = u["w"] * (u["xe"] - eff_s)
                        shear  += F
                        moment -= F * ((eff_s + u["xe"]) / 2 - xi)  # hogging → negative
            for v in uvls:
                span_v = v["xe"] - v["xs"]
                if span_v > 0 and xi < v["xe"]:
                    eff_s = max(xi, v["xs"])
                    if v["xe"] > eff_s:
                        t0    = (eff_s - v["xs"]) / span_v
                        ws2   = v["w0"] + (v["w1"] - v["w0"]) * t0
                        F     = (ws2 + v["w1"]) / 2 * (v["xe"] - eff_s)
                        dv    = ws2 + v["w1"]
                        cx_v  = (eff_s + (v["xe"] - eff_s) * (ws2 + 2*v["w1"]) / (3*dv)
                                 if dv > 1e-9 else (eff_s + v["xe"]) / 2)
                        shear  += F
                        moment -= F * (cx_v - xi)  # hogging → negative
            for m in moments:
                if xi <= m["x"]:
                    moment += m["M"]   # clockwise external moment to right → positive on right portion
        else:
            shear  = 0.0
            moment = 0.0

            # Add reaction RA at its actual position xA
            if xi >= xA:
                shear  += RA
                moment += RA * (xi - xA)

            # Add reaction RB at its actual position xB_pos
            if xi >= xB_pos:
                shear  += RB
                moment += RB * (xi - xB_pos)

            # Subtract applied loads (left-to-right)
            for pl in point_loads:
                if xi >= pl["x"]:
                    shear  -= pl["P"]
                    moment -= pl["P"] * (xi - pl["x"])
            for u in udls:
                if xi >= u["xs"]:
                    eff_e  = min(xi, u["xe"])
                    F      = u["w"] * (eff_e - u["xs"])
                    shear  -= F
                    moment -= F * (xi - (u["xs"] + eff_e) / 2)
            for v in uvls:
                span_v = v["xe"] - v["xs"]
                if span_v > 0 and xi >= v["xs"]:
                    eff_e = min(xi, v["xe"])
                    t1    = (eff_e - v["xs"]) / span_v
                    w_xi  = v["w0"] + (v["w1"] - v["w0"]) * t1
                    F     = (v["w0"] + w_xi) / 2 * (eff_e - v["xs"])
                    denom = v["w0"] + w_xi
                    cx    = (v["xs"] + (eff_e - v["xs"]) * (v["w0"] + 2 * w_xi) / (3 * denom)
                             if denom > 1e-9 else (v["xs"] + eff_e) / 2)
                    shear  -= F
                    moment -= F * (xi - cx)
            for m in moments:
                if xi >= m["x"]:
                    moment -= m["M"]

        V[j]     = shear
        M_arr[j] = moment

    # ── Compute total moment about xA (for display in Step 2) ──────
    total_M_xA = sum(pl["P"] * (pl["x"] - xA) for pl in point_loads)
    for _u in udls:
        _F = _u["w"] * (_u["xe"] - _u["xs"])
        total_M_xA += _F * (((_u["xs"] + _u["xe"]) / 2) - xA)
    for _v in uvls:
        _len = _v["xe"] - _v["xs"]
        if _len > 0:
            _F = (_v["w0"] + _v["w1"]) / 2 * _len
            _denom = _v["w0"] + _v["w1"]
            _cx = (_v["xs"] + _len * (_v["w0"] + 2 * _v["w1"]) / (3 * _denom)
                   if abs(_denom) > 1e-9 else _v["xs"] + _len / 2)
            total_M_xA += _F * (_cx - xA)
    total_M_xA += sum(m["M"] for m in moments)

    # ── Results ───────────────────────────────
    st.header("Results")

    # ── STEP 1: Given Data ────────────────────
    with st.expander("Step 1: Given Data", expanded=True):
        st.markdown(f"**Beam Type:** {beam_type} &nbsp;|&nbsp; **Length:** L = {L/length_factor_m:.4g} {length_unit}")
        st.markdown("**Applied Loads:**")
        if point_loads:
            st.markdown("*Point Loads:*")
            for i, pl in enumerate(point_loads):
                st.latex(f"P_{{{i+1}}} = {pl['P']/force_factor_kn:.4g}\\text{{ {force_unit} at }} x = {pl['x']/length_factor_m:.4g}\\text{{ {length_unit}}}")
        if udls:
            st.markdown("*Uniformly Distributed Loads:*")
            for i, u in enumerate(udls):
                length_u = u["xe"] - u["xs"]
                F_u = u["w"] * length_u
                cx_u = (u["xs"] + u["xe"]) / 2
                st.latex(
                    f"\\text{{UDL}}_{{{i+1}}}: w = {u['w']/(force_factor_kn/length_factor_m):.4g}\\text{{ {force_unit}/{length_unit}}},\\;"
                    f"x = {u['xs']/length_factor_m:.4g}\\text{{ to }}{u['xe']/length_factor_m:.4g}\\text{{ {length_unit}}},\\;"
                    f"\\text{{Resultant}} = {F_u/force_factor_kn:.4g}\\text{{ {force_unit} at centroid }}x = {cx_u/length_factor_m:.4g}\\text{{ {length_unit}}}"
                )
        if uvls:
            st.markdown("*Uniformly Varying Loads:*")
            for i, v in enumerate(uvls):
                length_v = v["xe"] - v["xs"]
                F_v = (v["w0"] + v["w1"]) / 2 * length_v
                denom_v = v["w0"] + v["w1"]
                cx_v = (v["xs"] + length_v * (v["w0"] + 2*v["w1"]) / (3*denom_v)
                        if abs(denom_v) > 1e-9 else v["xs"] + length_v / 2)
                st.latex(
                    f"\\text{{UVL}}_{{{i+1}}}: w_0={v['w0']/(force_factor_kn/length_factor_m):.4g},\\; w_1={v['w1']/(force_factor_kn/length_factor_m):.4g}\\text{{ {force_unit}/{length_unit}}},\\;"
                    f"x={v['xs']/length_factor_m:.4g}\\text{{ to }}{v['xe']/length_factor_m:.4g}\\text{{ {length_unit}}},\\;"
                    f"F = {F_v/force_factor_kn:.4g}\\text{{ {force_unit} at }}x={cx_v/length_factor_m:.4g}\\text{{ {length_unit}}}"
                )
        if moments:
            st.markdown("*Point Moments:*")
            for i, m in enumerate(moments):
                st.latex(f"M_{{{i+1}}} = {m['M']/(force_factor_kn*length_factor_m):.4g}\\text{{ {force_unit}\\cdot{length_unit} at }}x = {m['x']/length_factor_m:.4g}\\text{{ {length_unit}}}")

    # ── STEP 2: Equilibrium & Reactions ───────
    with st.expander("Step 2: Equilibrium Equations & Reactions", expanded=True):
        st.markdown("**Sum of vertical forces:**")
        st.latex(f"\\sum F_y = 0 \\implies R_A + R_B = {total_F/force_factor_kn:.3f}\\text{{ {force_unit}}}")

        xA_disp = xA / length_factor_m
        st.markdown(f"**Sum of moments about Support A (x = {xA_disp:.4g} {length_unit}):**")
        moment_terms = []
        for pl in point_loads:
            arm = (pl["x"] - xA) / length_factor_m
            moment_terms.append(f"{pl['P']/force_factor_kn:.4g} \\times {arm:.4g}")
        for u in udls:
            F_u = u["w"] * (u["xe"] - u["xs"])
            cx_u = (u["xs"] + u["xe"]) / 2
            arm_u = (cx_u - xA) / length_factor_m
            moment_terms.append(f"{F_u/force_factor_kn:.4g} \\times {arm_u:.4g}")
        for v in uvls:
            length_v = v["xe"] - v["xs"]
            if length_v > 0:
                F_v = (v["w0"] + v["w1"]) / 2 * length_v
                denom_v = v["w0"] + v["w1"]
                cx_v = (v["xs"] + length_v * (v["w0"] + 2*v["w1"]) / (3*denom_v)
                        if abs(denom_v) > 1e-9 else v["xs"] + length_v / 2)
                arm_v = (cx_v - xA) / length_factor_m
                moment_terms.append(f"{F_v/force_factor_kn:.4g} \\times {arm_v:.4g}")
        for m in moments:
            moment_terms.append(f"{m['M']/(force_factor_kn*length_factor_m):.4g}")

        if moment_terms:
            st.latex("\\sum M_A = 0 \\implies " + " + ".join(moment_terms)
                     + f" = {total_M_xA/(force_factor_kn*length_factor_m):.3f}\\text{{ {force_unit}\\cdot{length_unit}}}")

        if beam_type == "Simply Supported":
            st.markdown("**Taking moment about A to find R_B:**")
            span = xB_pos - xA
            mom_num = RB * span / (force_factor_kn * length_factor_m)
            st.latex(f"R_B \\times ({xB_pos/length_factor_m:.4g} - {xA/length_factor_m:.4g}) = {mom_num:.3f}")
            st.latex(f"R_B = \\frac{{{mom_num:.3f}}}{{{span/length_factor_m:.4g}}} = {RB/force_factor_kn:.3f}\\text{{ {force_unit}}}")
            st.latex(f"R_A = {total_F/force_factor_kn:.3f} - {RB/force_factor_kn:.3f} = {RA/force_factor_kn:.3f}\\text{{ {force_unit}}}")

        elif beam_type == "Cantilever (Fixed-Free)":
            st.markdown("**Fixed end reactions:**")
            st.latex(f"R_A = \\sum F_y = {RA/force_factor_kn:.3f}\\text{{ {force_unit} (upward)}}")
            st.latex(f"M_{{fixed}} = \\sum M_A = {MA_fix/(force_factor_kn*length_factor_m):.3f}\\text{{ {force_unit}\\cdot{length_unit}}}")

        elif beam_type in ["Overhanging (Left)", "Overhanging (Right)"]:
            span = xB_pos - xA
            st.markdown(f"**Support A at x={xA/length_factor_m:.4g} {length_unit}, Support B at x={xB_pos/length_factor_m:.4g} {length_unit}**")
            st.markdown("Taking moment about support A to find R_B:")
            mom_num = RB * span / (force_factor_kn * length_factor_m)
            st.latex(f"R_B \\times {span/length_factor_m:.4g} = {mom_num:.3f}")
            st.latex(f"R_B = \\frac{{{mom_num:.3f}}}{{{span/length_factor_m:.4g}}} = {RB/force_factor_kn:.3f}\\text{{ {force_unit}}}")
            st.latex(f"R_A = {total_F/force_factor_kn:.3f} - {RB/force_factor_kn:.3f} = {RA/force_factor_kn:.3f}\\text{{ {force_unit}}}")

        elif beam_type in ["Propped Cantilever", "Fixed-Fixed"]:
            st.markdown(f"**Support A at x={xA/length_factor_m:.4g} {length_unit}, Support B at x={xB_pos/length_factor_m:.4g} {length_unit}**")
            st.latex(f"R_A = {RA/force_factor_kn:.3f}\\text{{ {force_unit} (approx.)}},\\quad R_B = {RB/force_factor_kn:.3f}\\text{{ {force_unit} (approx.)}}")

        st.markdown("**Verification — Check equilibrium:**")
        check = RA + RB - total_F
        st.latex(
            f"R_A + R_B - \\sum F = {RA/force_factor_kn:.3f} + {RB/force_factor_kn:.3f} - "
            f"{total_F/force_factor_kn:.3f} = {check/force_factor_kn:.4f} \\approx 0 \\;\\checkmark"
        )

    # ── STEP 3: Shear Force Equations ─────────
    with st.expander("Step 3: Shear Force Equations (by section)", expanded=True):
        # Collect all critical x positions and sort
        critical_x = sorted(set(
            [0.0, L] +
            [xA, xB_pos] +
            [pl["x"] for pl in point_loads] +
            [u["xs"] for u in udls] + [u["xe"] for u in udls] +
            [v["xs"] for v in uvls] + [v["xe"] for v in uvls] +
            [m["x"] for m in moments]
        ))
        critical_x = [xi for xi in critical_x if 0.0 <= xi <= L]

        st.markdown("**Critical positions (x):** " + ", ".join(
            f"{xi/length_factor_m:.4g} {length_unit}" for xi in critical_x))
        st.markdown("")

        if beam_type != "Cantilever (Fixed-Free)":
            st.markdown("**Section-by-section shear (from left):**")
            prev_x = 0.0
            for xi_c in critical_x[1:]:
                # Compute V at the START of this section (at prev_x)
                v_before = 0.0
                if prev_x >= xA:
                    v_before += RA
                if prev_x >= xB_pos:
                    v_before += RB
                for pl in point_loads:
                    if prev_x >= pl["x"]: v_before -= pl["P"]
                for u in udls:
                    if prev_x > u["xs"]:
                        eff_e = min(prev_x, u["xe"])
                        v_before -= u["w"] * (eff_e - u["xs"])
                for v_l in uvls:
                    span_v = v_l["xe"] - v_l["xs"]
                    if span_v > 0 and prev_x > v_l["xs"]:
                        eff_e = min(prev_x, v_l["xe"])
                        t1 = (eff_e - v_l["xs"]) / span_v
                        w_xi = v_l["w0"] + (v_l["w1"] - v_l["w0"]) * t1
                        F = (v_l["w0"] + w_xi) / 2 * (eff_e - v_l["xs"])
                        v_before -= F

                # describe what's in this section (any overlap)
                section_desc = []
                for u in udls:
                    if u["xs"] < xi_c and u["xe"] > prev_x:
                        section_desc.append(f"UDL {u['w']/(force_factor_kn/length_factor_m):.4g} {force_unit}/{length_unit}")
                for v_l in uvls:
                    if v_l["xs"] < xi_c and v_l["xe"] > prev_x:
                        w0d = v_l['w0']/(force_factor_kn/length_factor_m)
                        w1d = v_l['w1']/(force_factor_kn/length_factor_m)
                        section_desc.append(f"UVL {w0d:.4g}→{w1d:.4g} {force_unit}/{length_unit}")

                desc_str = f"({', '.join(section_desc)})" if section_desc else "(no distributed load)"
                st.markdown(f"**Section: {prev_x/length_factor_m:.4g} – {xi_c/length_factor_m:.4g} {length_unit}** {desc_str}")
                st.latex(f"V = {v_before/force_factor_kn:.3f}\\text{{ {force_unit} (at start of section)}}")
                prev_x = xi_c
        else:
            st.markdown("**Cantilever: shear computed from free end (right) toward fixed end (left).**")

        st.markdown("**Shear values at key points:**")
        for xi_c in critical_x:
            idx = int(xi_c / L * (N - 1))
            idx = min(idx, N - 1)
            st.latex(f"V(x={xi_c/length_factor_m:.4g}\\text{{ {length_unit}}}) = {V[idx]/force_factor_kn:.3f}\\text{{ {force_unit}}}")

    # ── STEP 4: Bending Moment Equations ──────
    with st.expander("Step 4: Bending Moment Equations (by section)", expanded=True):
        st.markdown("**Bending moment values at key points:**")
        for xi_c in critical_x:
            idx = int(xi_c / L * (N - 1))
            idx = min(idx, N - 1)
            st.latex(
                f"M(x={xi_c / length_factor_m:.3f}\\text{{ {length_unit}}}) = "
                f"{M_arr[idx] / (force_factor_kn * length_factor_m):.3f}\\text{{ {force_unit}\N{MIDDLE DOT}{length_unit}}}"
            )

        # Find zero-shear points (max moment locations)
        zero_cross = []
        for j in range(N - 1):
            if V[j] * V[j+1] < 0:
                x_zero = x[j] - V[j] * (x[j+1] - x[j]) / (V[j+1] - V[j])
                zero_cross.append(x_zero)
        if zero_cross:
            st.markdown("**Zero shear (maximum moment) locations:**")
            for xz in zero_cross:
                idx_z = int(xz / L * (N - 1))
                idx_z = min(idx_z, N - 1)
                st.latex(
                    f"V = 0 \\text{{ at }} x = {xz / length_factor_m:.3f}\\text{{ {length_unit}}} \\implies "
                    f"M_{{max}} = {M_arr[idx_z] / (force_factor_kn * length_factor_m):.3f}\\text{{ {force_unit}\N{MIDDLE DOT}{length_unit}}}"
                )

    # ── STEP 5: Summary ───────────────────────
    with st.expander("Step 5: Summary of Results", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Shear V+", f"{np.max(V) / force_factor_kn:.2f} {force_unit}")
        c2.metric("Min Shear V-", f"{np.min(V) / force_factor_kn:.2f} {force_unit}")
        c3.metric("Max Moment M+", f"{np.max(M_arr) / (force_factor_kn * length_factor_m):.2f} {force_unit}·{length_unit}")
        c4.metric("Min Moment M-", f"{np.min(M_arr) / (force_factor_kn * length_factor_m):.2f} {force_unit}·{length_unit}")

        if beam_type == "Cantilever (Fixed-Free)":
            st.latex(
                f"R_A = {RA / force_factor_kn:.3f}\\text{{ {force_unit}}},\\quad "
                f"M_{{fixed}} = {MA_fix / (force_factor_kn * length_factor_m):.3f}\\text{{ {force_unit}\N{MIDDLE DOT}{length_unit}}}"
            )
        else:
            st.latex(
                f"R_A = {RA / force_factor_kn:.3f}\\text{{ {force_unit}}},\\quad "
                f"R_B = {RB / force_factor_kn:.3f}\\text{{ {force_unit}}}"
            )

    # ── SFD & BMD plots ───────────────────────
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    fig2.patch.set_facecolor("#0e1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    def find_peak_indices(y_vals):
        """Return indices of significant local maxima/minima plus global extrema."""
        peaks = []
        rng = float(np.max(y_vals) - np.min(y_vals))
        eps = max(1e-6, 0.005 * rng)
        n = len(y_vals)
        for i in range(1, n - 1):
            y0, y1, y2 = y_vals[i - 1], y_vals[i], y_vals[i + 1]
            is_local_max = (y1 >= y0 and y1 > y2) or (y1 > y0 and y1 >= y2)
            is_local_min = (y1 <= y0 and y1 < y2) or (y1 < y0 and y1 <= y2)
            if (is_local_max or is_local_min) and (abs(y1 - y0) > eps or abs(y1 - y2) > eps):
                peaks.append(i)

        peaks.extend([int(np.argmax(y_vals)), int(np.argmin(y_vals))])
        peaks = sorted(set(peaks))
        return peaks

    # ── Build key x positions for vertical lines ──────────────────
    x_plot = x / length_factor_m
    V_plot = V / force_factor_kn
    M_plot = M_arr / (force_factor_kn * length_factor_m)

    # Collect unique key positions in display units
    key_positions_m = sorted(set(
        [0.0, L, xA, xB_pos] +
        [pl["x"] for pl in point_loads] +
        [u["xs"] for u in udls] + [u["xe"] for u in udls] +
        [v["xs"] for v in uvls] + [v["xe"] for v in uvls] +
        [m["x"] for m in moments]
    ))
    key_positions_m = [xi for xi in key_positions_m if 0.0 <= xi <= L]
    key_positions_disp = [xi / length_factor_m for xi in key_positions_m]

    # Label color per position type
    def _pos_color(xi_m):
        if abs(xi_m - xA) < 1e-6 or abs(xi_m - xB_pos) < 1e-6:
            return "#f0a500"   # support — amber
        for pl in point_loads:
            if abs(xi_m - pl["x"]) < 1e-6:
                return "#ff4444"   # point load — red
        for u in udls:
            if abs(xi_m - u["xs"]) < 1e-6 or abs(xi_m - u["xe"]) < 1e-6:
                return "#00bfff"   # UDL — blue
        for v in uvls:
            if abs(xi_m - v["xs"]) < 1e-6 or abs(xi_m - v["xe"]) < 1e-6:
                return "#ff6eb4"   # UVL — pink
        for m in moments:
            if abs(xi_m - m["x"]) < 1e-6:
                return "#ffd700"   # moment — gold
        return "#666666"           # beam ends — grey

    def _draw_vlines(ax, y_vals):
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        y_range = y_max - y_min if abs(y_max - y_min) > 1e-9 else 1.0
        # Draw vertical lines and distance labels at bottom
        prev_disp = None
        for xi_m, xi_d in zip(key_positions_m, key_positions_disp):
            col = _pos_color(xi_m)
            ax.axvline(xi_d, color=col, lw=0.9, ls="--", alpha=0.55)
            # position label at top of axes
            ax.text(xi_d, y_max + y_range * 0.04,
                    f"{xi_d:.3g}",
                    ha="center", va="bottom", color=col, fontsize=7,
                    rotation=90, clip_on=False)
            # distance arrow between consecutive key positions
            if prev_disp is not None:
                dist = xi_d - prev_disp
                mid = (xi_d + prev_disp) / 2
                ax.annotate(
                    "", xy=(xi_d, y_min - y_range * 0.18),
                    xytext=(prev_disp, y_min - y_range * 0.18),
                    arrowprops=dict(arrowstyle="<->", color="#aaa", lw=0.8),
                    annotation_clip=False,
                )
                ax.text(mid, y_min - y_range * 0.25,
                        f"{dist:.3g} {length_unit}",
                        ha="center", va="top", color="#aaa", fontsize=6.5,
                        clip_on=False)
            prev_disp = xi_d

    # SFD
    ax1.set_title("Shear Force Diagram (SFD)", fontsize=13, fontweight="bold")
    ax1.fill_between(x_plot, V_plot, 0, where=(V_plot >= 0), alpha=0.45, color="#00bfff", label="Positive")
    ax1.fill_between(x_plot, V_plot, 0, where=(V_plot < 0),  alpha=0.45, color="#ff6666", label="Negative")
    ax1.plot(x_plot, V_plot, color="white", lw=1.8)
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.set_xlabel(f"x ({length_unit})")
    ax1.set_ylabel(f"V ({force_unit})")
    ax1.legend(facecolor="#1a1d23", labelcolor="white", fontsize=8)
    _draw_vlines(ax1, V_plot)
    sfd_peaks = find_peak_indices(V_plot)
    for k, idx in enumerate(sfd_peaks):
        y_off = 6 if k % 2 == 0 else -12
        color = "#00ffff" if V_plot[idx] >= 0 else "#ff9999"
        ax1.annotate(
            f"{V_plot[idx]:.2f}",
            xy=(x_plot[idx], V_plot[idx]),
            color=color,
            fontsize=8,
            xytext=(4, y_off),
            textcoords="offset points"
        )

    # BMD
    ax2.set_title("Bending Moment Diagram (BMD)", fontsize=13, fontweight="bold")
    ax2.fill_between(x_plot, M_plot, 0, where=(M_plot >= 0), alpha=0.45, color="#50c878", label="Sagging (+)")
    ax2.fill_between(x_plot, M_plot, 0, where=(M_plot < 0),  alpha=0.45, color="#ff8c00", label="Hogging (–)")
    ax2.plot(x_plot, M_plot, color="white", lw=1.8)
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_xlabel(f"x ({length_unit})")
    ax2.set_ylabel(f"M ({force_unit}·{length_unit})")
    ax2.legend(facecolor="#1a1d23", labelcolor="white", fontsize=8)
    _draw_vlines(ax2, M_plot)
    bmd_peaks = find_peak_indices(M_plot)
    for k, idx in enumerate(bmd_peaks):
        y_off = 6 if k % 2 == 0 else -12
        color = "#90ee90" if M_plot[idx] >= 0 else "#ffb347"
        ax2.annotate(
            f"{M_plot[idx]:.2f}",
            xy=(x_plot[idx], M_plot[idx]),
            color=color,
            fontsize=8,
            xytext=(4, y_off),
            textcoords="offset points"
        )

    plt.tight_layout(pad=2)
    st.pyplot(fig2, width='stretch')
    plt.close(fig2)

