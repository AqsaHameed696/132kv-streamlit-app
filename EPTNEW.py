# streamlit_132kV_full_app_updated_fixed.py
# Fixed plotting for Geo1 and Geo2 comparison charts (normalised improvement ratios)
# Place background MP4 in same folder and name it 'background.mp4'

import streamlit as st
import math, cmath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='132 kV Line — Single vs Bundle (3 tabs)', layout='wide')
st.title ("132KV TRANSMISSION LINE PERFORMANCE ANALYZER AND STUDY IMPACT ON ENVIRONMENT")

# ---------------------------
# SIDEBAR — FIXED LINE CONSTANTS
# ---------------------------
with st.sidebar:
    st.header("Fixed Line Parameters")
    st.write("**Voltage Level:** 132 kV")
    st.write("**System Frequency:** 50 Hz")
    st.write("**Line Length:** 100 km")

# --------------------
# Basic conductor database
# --------------------
CONDUCTORS = {
    "Gopher":   {"diameter_mm": 7.08,  "R20": 1.093, "weight_kg_per_km":106.0,  "UTS_kg":980.0,   "area_mm2":30.62},
    "Rabbit":   {"diameter_mm": 10.05, "R20": 0.543, "weight_kg_per_km":214.0,  "UTS_kg":1875.0,  "area_mm2":61.69},
    "Dog":      {"diameter_mm": 14.15, "R20": 0.273, "weight_kg_per_km":394.0,  "UTS_kg":3225.0,  "area_mm2":118.53},
    "Lynx":     {"diameter_mm": 19.53, "R20": 0.158, "weight_kg_per_km":842.0,  "UTS_kg":7890.0,  "area_mm2":226.20},
    "Cuckoo":   {"diameter_mm": 27.72, "R20": 0.072, "weight_kg_per_km":1519.0, "UTS_kg":12385.0, "area_mm2":454.48},
    "Rail":     {"diameter_mm": 29.61, "R20": 0.060, "weight_kg_per_km":1599.0, "UTS_kg":11874.0, "area_mm2":517.38},
    "Cardinal": {"diameter_mm": 30.42, "R20": 0.060, "weight_kg_per_km":1832.0, "UTS_kg":15262.0, "area_mm2":547.30},

    # Added Conductors
    "Martin":   {"diameter_mm": 17.35, "R20": 0.1691, "weight_kg_per_km":544.0,   "UTS_kg":21000.0, "area_mm2":170.5},
    "Panther":  {"diameter_mm": 21.00, "R20": 0.1363, "weight_kg_per_km":974.0,   "UTS_kg":9400.0,  "area_mm2":212.1},
    "Osprey":   {"diameter_mm": 22.35, "R20": 0.1022, "weight_kg_per_km":898.8,   "UTS_kg":5960.0,  "area_mm2":281.9},
    "Zebra":    {"diameter_mm": 28.62, "R20": 0.0680, "weight_kg_per_km":1350.0,  "UTS_kg":13190.0, "area_mm2":428.0}
}


# --------------------
# physical constants + helper functions
# --------------------
mu0 = 4*math.pi*1e-7
eps0 = 8.854e-12
alpha_resist = 0.00403
g = 9.80665
alpha_air = 17.73e-6

def kgpkm_to_Npm(kgpkm): return (kgpkm/1000.0)*g
def area_mm2_to_m2(a): return a*1e-6
def wind_load_on_cylinder(d_m, wind_pressure=390.0): return wind_pressure * d_m * (2.0/3.0)
def sag_parabolic(w, L, T): 
    if T<=0: return None
    return (w * L**2)/(8.0 * T)

# --------------------
# Corona (Peek's formula approximate)
# --------------------
def peek_corona_loss(Vr_line, Vr_phase, r_m, GMD_m, f=50, pressure=76, temp_C=20, m0=0.85):
    r_cm = r_m*100.0
    R_cm = GMD_m*100.0
    delta = (3.92 * pressure) / (273 + temp_C)
    Vmax_kV = (Vr_phase/1000.0) * math.sqrt(2)
    Vd = 21.1 * m0 * delta * r_cm * math.log(GMD_m / r_m + 1e-12)
    if Vmax_kV <= Vd:
        return 0.0
    corona_w = (242.4/delta)*(f+25)*math.sqrt(r_cm/R_cm)*(Vmax_kV - Vd)**2 * 1000.0
    return max(0.0, corona_w)

# --------------------
# Main electrical + mechanical compute (single circuit)
# --------------------
def compute_params(conductor_key, Dab, Dbc, Dca, span_m, show_bundle=False, S_bundle=0.4,
                   Vr_line=132e3, L_m=100e3, P_load=50e6, pf_load=0.9, f=50, double_circuit=False):
    info = CONDUCTORS[conductor_key]
    d = info['diameter_mm']/1000.0
    r = d/2.0
    R20 = info['R20']
    R75 = R20 * (1 + alpha_resist*(75-20))
    R_total = R75 * (L_m/1000.0)

    GMD = (Dab * Dbc * Dca)**(1/3)

    if not show_bundle:
        GMR_L = 0.7788 * r
        Req = r
    else:
        GMR_sub = 0.7788 * r
        GMR_L = math.sqrt(GMR_sub * S_bundle)
        Req = math.sqrt(r * S_bundle)

    omega = 2*math.pi*f
    Lp = mu0/(2*math.pi) * math.log(GMD/(GMR_L+1e-12))
    L_total = Lp * L_m
    Cp = 2*math.pi*eps0 / math.log(GMD/(Req+1e-12))
    C_total = Cp * L_m

    Z = complex(R_total, omega * L_total)
    Y = complex(0, omega * C_total)

    A = 1 + (Z*Y)/2
    D = A
    B = Z*(1 + (Z*Y)/4)
    C = Y*(1 + (Z*Y)/4)

    Vr_phase = Vr_line / math.sqrt(3)
    Ir_mag = P_load / (math.sqrt(3) * Vr_line * pf_load)
    Ir_angle = -math.acos(pf_load)
    Ir = cmath.rect(Ir_mag, Ir_angle)

    Vs_phase = A*Vr_phase + B*Ir
    Is_phase = C*Vr_phase + D*Ir

    Vs_line_kV = abs(Vs_phase) * math.sqrt(3) / 1000.0
    Is_line_A = abs(Is_phase)

    Vr_line_kV = Vr_line / 1000.0
    Voltage_Reg = ((Vs_line_kV - Vr_line_kV) / Vr_line_kV) * 100.0

    S_sent = 3 * Vs_phase * Is_phase.conjugate()
    P_sent = S_sent.real
    efficiency = (P_load / P_sent) * 100.0 if P_sent != 0 else None

    corona_w_per_km = peek_corona_loss(Vr_line, Vr_phase, r, GMD, f=f)

    multiplier = 2.0 if double_circuit else 1.0

    W_self = kgpkm_to_Npm(info['weight_kg_per_km'])
    wind = wind_load_on_cylinder(d)
    UTS = info['UTS_kg']
    T_work = (UTS / 2.5) * g
    W_total = math.sqrt(W_self**2 + wind**2)
    sag_m = sag_parabolic(W_total, span_m, T_work)

    return {
        "Conductor": conductor_key,
        "Config": "Bundle-2" if show_bundle else "Single",
        "Vs_line_kV": Vs_line_kV,
        "Is_line_A": Is_line_A,
        "Efficiency_%": efficiency,
        "Voltage_Reg_%": Voltage_Reg,
        "Corona_W_per_km": corona_w_per_km * multiplier,
        "R_total_ohm": R_total,
        "L_total_H": L_total,
        "C_total_F": C_total,
        "Sag_m": sag_m
    }

# --------------------
# Utility: compute improvement ratios safely
# --------------------
def improvement_ratios(single_dict, bundle_dict, metrics, higher_is_better):
    # returns ratios where >1 means bundle improved over single
    ratios = []
    abs_single = []
    abs_bundle = []
    for m in metrics:
        s = single_dict.get(m, np.nan)
        b = bundle_dict.get(m, np.nan) if bundle_dict is not None else np.nan
        abs_single.append(s)
        abs_bundle.append(b)
        # handle NaN or zero
        try:
            s_val = float(s) if s is not None else np.nan
            b_val = float(b) if b is not None else np.nan
        except Exception:
            s_val = np.nan
            b_val = np.nan
        if np.isnan(s_val) or np.isnan(b_val) or s_val == 0 or b_val == 0:
            ratios.append(np.nan)
            continue
        if higher_is_better.get(m, False):
            # higher better: ratio = bundle / single
            ratios.append(b_val / s_val)
        else:
            # lower better: ratio = single / bundle
            ratios.append(s_val / b_val)
    return np.array(ratios, dtype=float), np.array(abs_single, dtype=float), np.array(abs_bundle, dtype=float)

# --------------------
# Background video (fixed full-screen)
# --------------------
bg_video_filename = "17722621-uhd_2160_3840_24fps.mp4"
bg_html = f"""
<style>
.video-bg {{
  position: fixed;
  right: 0;
  bottom: 0;
  min-width: 100%;
  min-height: 100%;
  z-index: -1;
  opacity: 0.12;
}}
.container-main {{
  position: relative;
  z-index: 2;
}}
</style>
<video autoplay muted loop class="video-bg">
  <source src="{bg_video_filename}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
st.markdown(bg_html, unsafe_allow_html=True)

# --------------------
# Layout: Tabs
# --------------------
tab1, tab2, tab3 = st.tabs(["Geo1 — Delta (Single Ckt)", "Geo2 — Vertical (Double Ckt)", "Environmental Impact & Papers"])

# Metrics and direction map: True if higher value is better
metrics = ["Vs_line_kV", "Is_line_A", "Efficiency_%", "Voltage_Reg_%", "Corona_W_per_km", "Sag_m"]
# For our interpretation: Efficiency higher better; Vs_line close to nominal -> we treat deviation implicitly, but for ratio plotting we'll treat Vs_line_kV higher closer to nominal as 'better'.
higher_is_better = {
    "Vs_line_kV": True,
    "Is_line_A": False,       # lower current better
    "Efficiency_%": True,
    "Voltage_Reg_%": False,   # lower regulation better
    "Corona_W_per_km": False, # lower corona loss better
    "Sag_m": False            # lower sag better
}

# --------------------
# TAB 1: Geo1 (updated layout like Tab2, stringing physics for mech/elect)
# --------------------
with tab1:
    st.header("Geo1 — Delta shape (Single circuit). Improvement: bundle (2 subconductors)")
    st.image("tower.jpg", caption="132 kV Single-Circuit Tower", use_container_width= True)
    st.image("tower_label.PNG", caption="132 kV Single-Circuit Tower Labeled Daigram", use_container_width= True)

    col1, col2 = st.columns(2)

    with col1:
        conductor_choice = st.selectbox("Select conductor (Geo1)", list(CONDUCTORS.keys()), index=0)
        Dab = st.number_input("Dab (m)", value=5.95)
        Dbc = st.number_input("Dbc (m)", value=5.95)
        Dca = st.number_input("Dca (m)", value=8.08)
        span_m = st.selectbox("Span (m)", [200,250,275,300,325], index=0)

    with col2:
        pf_load = st.number_input("Receiving PF", value=0.9, min_value=0.5, max_value=1.0, step=0.01)
        P_load = st.number_input("Receiving Active Power P (W)", value=100e6, format="%.0f")
        S_bundle = st.number_input("Bundle spacing (m) if improved", value=0.45)
        show_bundle = st.checkbox("Show bundle improvement (Geo1)")

    # Electrical results for selected conductor (single / bundle rows)
    out_single = compute_params(conductor_choice, Dab, Dbc, Dca, span_m, show_bundle=False, S_bundle=S_bundle)
    out_bundle = compute_params(conductor_choice, Dab, Dbc, Dca, span_m, show_bundle=show_bundle, S_bundle=S_bundle) if show_bundle else None

    st.subheader("Electrical results — Selected conductor (Geo1)")
    df_elec = pd.DataFrame([out_single] + ([out_bundle] if out_bundle else []))
    st.dataframe(df_elec[['Conductor','Config','Vs_line_kV','Is_line_A','Efficiency_%','Voltage_Reg_%','Corona_W_per_km','Sag_m']])

    # -----------------------------
    # Graph Comparison — Before vs After Improvement (Geo1) — FIXED
    # -----------------------------
    st.subheader("Graph Comparison — Before vs After Improvement (Geo1)")

    if not show_bundle:
        st.info("Turn ON 'Show bundle improvement (Geo1)' to compare Single vs Bundle.")
    else:
        ratios, abs_s, abs_b = improvement_ratios(out_single, out_bundle, metrics, higher_is_better)

        fig, ax = plt.subplots(figsize=(10,5))
        x = np.arange(len(metrics))
        # Plot improvement ratio bars (>1 means bundle improved)
        ax.bar(x, np.nan_to_num(ratios, 1.0), width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylabel('Improvement ratio (bundle / single) — >1 means improvement')
        ax.set_title('Normalized Improvement — Geo1 (Bundle vs Single)')
        # annotate percent change
        for i, val in enumerate(ratios):
            if np.isnan(val):
                txt = 'N/A'
            else:
                pct = (val - 1.0) * 100.0
                txt = f"{pct:+.1f}%"
            ax.text(i, np.nan_to_num(val,1.0) + 0.02, txt, ha='center')
        st.pyplot(fig)

        # Also show absolute values (normalized to single baseline = 1) for visual comparison
        norm_b = abs_b / abs_s
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.bar(x - 0.15, np.ones_like(x), width=0.3, label='Single (baseline=1)')
        ax2.bar(x + 0.15, np.nan_to_num(norm_b, 1.0), width=0.3, label='Bundle (relative)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.set_ylabel('Relative to Single (1.0)')
        ax2.set_title('Absolute values normalized to Single baseline — Geo1')
        ax2.legend()
        st.pyplot(fig2)

# --------------------
# TAB 2: Geo2 - replaced with FULL double-circuit model
# --------------------
with tab2:
    st.header("Geo2 — Vertical configuration (Double circuit)")
    st.image("tower2.jpg", caption="132 kV Double-Circuit Tower", use_container_width=True)
    st.image("tower2_label.jpg", caption="132 kV Double-Circuit Tower Labeled Daigram", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        conductor_choice2 = st.selectbox("Select conductor (Geo2)", list(CONDUCTORS.keys()), index=0, key="g2")
        D_top_mid = st.number_input("Top-Mid spacing (m)", value=1.5, key="d1")
        D_mid_bottom = st.number_input("Mid-Bottom spacing (m)", value=1.5, key="d2")
        span_m2 = st.selectbox("Span (m)", [200,250,275,300,325], index=0, key="s2")

    with col2:
        pf_load2 = st.number_input("Receiving PF (Geo2)", value=0.9, min_value=0.5, max_value=1.0, step=0.01, key="pf2")
        P_load2 = st.number_input("Receiving Active Power P (W) (Geo2)", value=100e6, format="%.0f", key="p2")
        S_bundle2 = st.number_input("Bundle spacing (m) if improved (Geo2)", value=0.45, key="sb2")
        show_bundle2 = st.checkbox("Show bundle improvement (Geo2)", key="b2")

    Dab2 = D_top_mid
    Dbc2 = D_mid_bottom
    Dca2 = D_top_mid + D_mid_bottom

    out_single2 = compute_params(conductor_choice2, Dab2, Dbc2, Dca2, span_m2,
                                 show_bundle=False, S_bundle=S_bundle2, double_circuit=True)

    out_bundle2 = compute_params(conductor_choice2, Dab2, Dbc2, Dca2, span_m2,
                                 show_bundle=show_bundle2, S_bundle=S_bundle2, double_circuit=True) if show_bundle2 else None

    st.subheader("Electrical results — Selected conductor (Geo2)")
    df_elec2 = pd.DataFrame([out_single2] + ([out_bundle2] if out_bundle2 else []))
    st.dataframe(df_elec2[['Conductor','Config','Vs_line_kV','Is_line_A','Efficiency_%','Voltage_Reg_%','Corona_W_per_km','Sag_m']])

    st.subheader("Graph Comparison — Before vs After Improvement (Geo2)")
    if not show_bundle2:
        st.info("Enable 'Show bundle improvement (Geo2)' to compare Single vs Bundle.")
    else:
        ratios2, abs_s2, abs_b2 = improvement_ratios(out_single2, out_bundle2, metrics, higher_is_better)
        figb, axb = plt.subplots(figsize=(10,5))
        x2 = np.arange(len(metrics))
        axb.bar(x2, np.nan_to_num(ratios2,1.0), width=0.6)
        axb.set_xticks(x2)
        axb.set_xticklabels(metrics, rotation=45)
        axb.set_ylabel('Improvement ratio (bundle / single) — >1 means improvement')
        axb.set_title('Normalized Improvement — Geo2 (Bundle vs Single)')
        for i, val in enumerate(ratios2):
            if np.isnan(val): txt = 'N/A'
            else: txt = f"{(val-1.0)*100:+.1f}%"
            axb.text(i, np.nan_to_num(val,1.0) + 0.02, txt, ha='center')
        st.pyplot(figb)

        # Absolute normalized chart
        norm_b2 = abs_b2 / abs_s2
        figb2, axb2 = plt.subplots(figsize=(10,5))
        axb2.bar(x2 - 0.15, np.ones_like(x2), width=0.3, label='Single (baseline=1)')
        axb2.bar(x2 + 0.15, np.nan_to_num(norm_b2,1.0), width=0.3, label='Bundle (relative)')
        axb2.set_xticks(x2)
        axb2.set_xticklabels(metrics, rotation=45)
        axb2.set_ylabel('Relative to Single (1.0)')
        axb2.set_title('Absolute values normalized to Single baseline — Geo2')
        axb2.legend()
        st.pyplot(figb2)

# --------------------
# TAB 3: Environmental Impact & Research Links (unchanged)
# --------------------
with tab3:
    st.header("Environmental Impact — Summary & Research Links")
    st.write("Click each topic to expand details (slide-down style).")

    with st.expander("Corona: Audible noise, ozone and air effects"):
        st.write("""
        - Corona produces audible noise and ozone under certain weather conditions.  
        - Local communities may perceive noise; ozone generation is small but measurable near very high-voltage lines.  
        - Mitigation: choose bundled conductors, larger diameter conductors, and smoother surfaces to reduce corona inception.
        """)
    with st.expander("Avian (Bird) impacts"):
        st.write("""
        - Birds can collide with lines or be electrocuted on low-voltage distribution networks.  
        - For high-voltage transmission lines avoid siting in dense migratory paths, use bird diverters, visibility marking, perch deterrents where needed.
        - Reference guidance exists for bird-friendly design and mitigation measures.
        """)
    with st.expander("Visual landscape & land use"):
        st.write("""
        - Transmission corridors change landscape visibility and require right-of-way; consider routing, tower design, and vegetation management.
        - Environmental Impact Assessments (EIA) should examine habitat fragmentation.
        """)
    with st.expander("EMI & human health (brief)"):
        st.write("""
        - Corona and line EM fields produce electromagnetic fields (EMF). International guidelines exist for occupational exposure; typical transmission lines are within allowable limits at public distances.
        """)

    st.subheader("Selected Research Papers & References (useful reading)")
    st.markdown("[The Corona Phenomenon in Overhead Lines — MDPI (2021)](https://www.mdpi.com/1996-1073/14/20/6612)")
    st.markdown("[IET: A new method to calculate corona losses](https://digital-library.theiet.org/doi/full/10.1049/tje2.12155)")

st.markdown("---")
st.caption("Notes: Corona calculations are empirical approximations (Peek-style). Results are for comparison & screening; for final design use detailed IEC/IEEE calculations and field tests.")

# End of file
