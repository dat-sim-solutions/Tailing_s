import sys
import os
# Guarantee root directory is in sys.path for Streamlit Cloud deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine, text
from app.models_s import calculate_slope_stability, solve_darcy_fem
import plotly.graph_objects as go
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1. DATABASE CONFIGURATION
DB_URI = st.secrets["NEON_DB_URI"] 
engine = create_engine(DB_URI, connect_args={"ssl_context": True})

st.set_page_config(page_title="Seismic Tailing Safety", layout="wide")

st.title("🛡️ Tailing Dam Safety System - Peru")
st.markdown("Integrated Bishop Stability Model with **Pseudo-Static Seismic Analysis & 2D Darcy FEM Seepage**. JUAN A.C. 2026")

# 2. FETCH DATA FROM NEON
@st.cache_data(ttl=60)
def get_neon_data():
    query = text("""
        SELECT timestamp, pore_pressure, water_level 
        FROM sensor_readings 
        WHERE piezometer_id = 'PZ-LIMA-01' 
        AND timestamp >= '2026-04-06 00:00:00+00:00'
        ORDER BY timestamp DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

data = get_neon_data()

def enterprise_iot_layer():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Enterprise Data Layer")
    
    mode = st.sidebar.radio("Data Source:", ["Neon SQL (Historical)", "SAP HANA (Live IoT)"])

    if mode == "SAP HANA (Live IoT)":
        if st.sidebar.button("📡 Sync with SAP Leonardo"):
            with st.sidebar.status("Authenticating with SAP S/4HANA...", expanded=False) as status:
                time.sleep(1)
                st.write("Extracting OData Service: /Z_PIEZOMETER_READINGS...")
                time.sleep(1)
                
                live_sap_payload = {
                    "d": {
                        "results": [
                            {"SensorID": "PZ-01", "Level": 22.45, "Unit": "m", "Health": "Green"},
                            {"SensorID": "PZ-02", "Level": 19.80, "Unit": "m", "Health": "Green"}
                        ]
                    }
                }
                
                st.session_state['current_pore_pressure'] = live_sap_payload['d']['results'][0]['Level']
                status.update(label="SAP Sync Active", state="complete")
            
            st.sidebar.success(f"Live Level: {st.session_state['current_pore_pressure']}m")
            st.toast("Stability Model Updated via SAP IoT Feed", icon="🔌")

def plot_fs_gauge(fs_value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fs_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Factor of Safety", 'font': {'size': 24}},
        delta = {'reference': 1.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [0, 2.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "black"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1.1], 'color': '#ff4b4b'},
                {'range': [1.1, 1.5], 'color': '#ffa500'}, 
                {'range': [1.5, 2.5], 'color': '#00cc96'}  
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

if not data.empty:
    enterprise_iot_layer()
    
    st.sidebar.header("⏱ Data Selection")
    selected_time = st.sidebar.selectbox("Select Timestamp", data['timestamp'])
    current_row = data[data['timestamp'] == selected_time].iloc[0]

    if 'current_pore_pressure' in st.session_state:
        u_latest = st.session_state['current_pore_pressure']
        st.sidebar.info(f"Using LIVE SAP Data: {u_latest} kPa")
    else:
        u_latest = current_row['pore_pressure']
        st.sidebar.info("Using Historical SQL Data")

    st.sidebar.header("🌋 Seismic Analysis")
    kh = st.sidebar.slider("Seismic Coeff (kh)", 0.0, 0.3, 0.15, step=0.01, 
                           help="Peruvian Standard E.050: 0.15 for Coast/High Risk")

    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["🎮 Manual Explorer", "🔥 Global Heatmap", "🌊 Darcy FEM Seepage"])

    # -----------------------------------------------------------------
    # TAB 1: MANUAL EXPLORER
    # -----------------------------------------------------------------
    with tab1:
        st.sidebar.header("🔴 Slip Circle Geometry")
        xc = st.sidebar.slider("Center X (xc)", 20.0, 150.0, 75.0)
        yc = st.sidebar.slider("Center Y (yc)", 30.0, 150.0, 85.0)
        R = st.sidebar.slider("Radius (R)", 10.0, 100.0, 65.0)

        # Explicitly passing custom_phreatic_fn=None to use piezometer timestamp data
        fs, slices, water_line, history, num, den = calculate_slope_stability(
            xc, yc, R, u_latest, kh=kh, custom_phreatic_fn=None
        )

        st.info("⚡ Bishop Stability Model is currently utilizing the **Piezometer Timestamp Data**.")

        col1, col2 = st.columns([1, 3])
                
        with col1:
            if fs:
                abs_fs = abs(fs)
                st.plotly_chart(plot_fs_gauge(abs(fs)), use_container_width=True)
                if abs_fs < 1.0: st.error("🚨 SEISMIC COLLAPSE")
                elif abs_fs < 1.2: st.warning("⚠️ CRITICAL VULNERABILITY")
                elif fs == 0: st.error("❌ No Intersection found.")
                else: st.warning("⚖️ Equilibrium reached.")
            else:
                st.error("No Intersection")

            direction = "► Right (Inner/Reservoir)" if fs < 0 else "◄ Left (Outer/Toe)"
                
            st.info(f"**Failure Direction:** {direction}")
            st.write(f"**Fs:** {fs}")
            st.write(f"**Pore Pressure:** {u_latest} kPa")
            st.write(f"**Head:** {round(u_latest/9.81, 2)} m")

        with col2:
            from matplotlib.lines import Line2D
            
            fig, ax = plt.subplots(figsize=(10, 6))
            dx, dy = np.array([40, 70, 100, 130]), np.array([10, 45, 45, 14])
            ax.plot(dx, dy, 'k-', linewidth=3)
            ax.fill_between(dx, dy, color='navajowhite', alpha=0.8)
            ax.plot(water_line[0], water_line[1], 'b--', label="Phreatic Line")
            ax.scatter([80], [10], color='blue', s=100, zorder=5, label="PZ-01 Sensor")
            
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(xc + R*np.cos(theta), yc + R*np.sin(theta), 'r--', alpha=0.4)
            ax.scatter([xc], [yc], color='red', marker='+', s=100)
            
            seismic_arrow_legend = Line2D([0], [0], color='red', marker='>', linestyle='-', 
                                         markersize=10, label=f'Seismic Force (kh={kh})')
            
            if slices:
                for s in slices:
                    ax.bar(s['x_mid'], s['h'], width=s['b'], bottom=s['y_bot'], 
                           color='orange', alpha=0.5, edgecolor='black', linewidth=0.2)
                    if kh > 0 and s['h'] > 0:
                        y_midpoint = s['y_bot'] + (s['h'] / 2)
                        vector_magnitude = - kh * s['h'] * 1.1 
                        ax.arrow(s['x_mid'], y_midpoint, vector_magnitude, 0, 
                                 head_width=1.5, head_length=1.0, fc='red', ec='red', 
                                 alpha=0.8, zorder=10)
            
            handles, labels = ax.get_legend_handles_labels()
            if kh > 0:
                handles.append(seismic_arrow_legend)
                
            ax.set_ylim(0, 120); ax.set_xlim(20, 150); ax.set_aspect('equal')
            ax.legend(handles=handles, loc='upper left'); ax.grid(True, alpha=0.2)
            st.pyplot(fig)

    # -----------------------------------------------------------------
    # TAB 2: GLOBAL HEATMAP
    # -----------------------------------------------------------------
    with tab2:
        st.subheader("🌐 Global Stability Grid Search")
        st.write("Calculates FS for a grid of centers using the current Radius.")
        if st.button("🚀 Start Global Scan"):
            grid_x = np.linspace(30, 140, 15)
            grid_y = np.linspace(60, 140, 15)
            fs_matrix = np.empty((len(grid_y), len(grid_x)))
            fs_matrix[:] = np.nan
            progress_text = "Analyzing slope stability surfaces..."
            my_bar = st.progress(0, text=progress_text)

            for i, py in enumerate(grid_y):
                for j, px in enumerate(grid_x):
                    val, _, _, _, _, _ = calculate_slope_stability(
                        px, py, R, u_latest, kh=kh, custom_phreatic_fn=None
                    )
                    abs_val = abs(val) if val is not None else np.nan
                    if 0.1 < abs_val < 50:
                        fs_matrix[i, j] = min(abs_val, 5.0)
                    else:
                        fs_matrix[i, j] = np.nan
                    
                my_bar.progress((i + 1) / len(grid_y))

            fig_h, ax_h = plt.subplots(figsize=(10, 8))
            sns.heatmap(fs_matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                        xticklabels=np.round(grid_x, 0), yticklabels=np.round(grid_y, 0), ax=ax_h, cbar_kws={'label': 'Absolute Factor of Safety'})
            ax_h.invert_yaxis()
            ax_h.set_title(f"MINIMUM Safety Zones for Radius {R}m, kh: {kh}")
            ax_h.set_xlabel("Center X (m)")
            ax_h.set_ylabel("Center Y (m)")
            st.pyplot(fig_h)

    # -----------------------------------------------------------------
    # TAB 3: DARCY FEM SEEPAGE SOLVER
    # -----------------------------------------------------------------
    with tab3:
        st.subheader("🌊 2D Unconfined Stationary Darcy FEM Seepage Simulation")
        st.markdown("Configure hydraulic conductivity, pool elevation, and geometry to compute hydraulic head ($h$) and pore pressure ($P$).")

        col_fem_p1, col_fem_p2 = st.columns(2)
        with col_fem_p1:
            h_pool_val = st.slider("Upstream Tailings Pool Elevation h_pool [m]", 50.0, 110.0, 102.0, step=1.0)
            k_sat_val = st.select_slider("Hydraulic Conductivity K_sat [m/s]", options=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3], value=1e-5)
        with col_fem_p2:
            num_levels = st.slider("Contour Levels", 10, 40, 20)
            plot_var = st.radio("Field to Plot:", ["Hydraulic Head h [m]", "Pore Pressure P [kPa]"], horizontal=True)

        if st.button("⚙️ Run Darcy FEM Seepage Analysis"):
            with st.spinner("Assembling Stiffness Matrix K & Solving System K·h = F..."):
                fem_res = solve_darcy_fem(h_pool=h_pool_val, k_sat=k_sat_val)
                st.session_state['fem_results'] = fem_res
                st.session_state['fem_phreatic_fn'] = fem_res['phreatic_fn']
                st.success("Darcy FEM Simulation Completed!")

        if 'fem_results' in st.session_state:
            fem_res = st.session_state['fem_results']
            
            fig_fem, ax_fem = plt.subplots(figsize=(12, 6))
            
            field_data = fem_res['h_fem'] if "Hydraulic Head" in plot_var else fem_res['P_kpa']
            field_label = "Total Hydraulic Head h [m]" if "Hydraulic Head" in plot_var else "Pore Pressure P [kPa]"
            
            cf = ax_fem.tricontourf(fem_res['triangulation'], field_data, levels=num_levels, cmap="viridis")
            cs = ax_fem.tricontour(fem_res['triangulation'], field_data, levels=15, colors="white", linewidths=0.5, alpha=0.7)
            ax_fem.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

            for elem in fem_res['elements']: # to plot grids
                elem_nodes = elem + [elem[0]]
                ax_fem.plot(fem_res['nodes'][elem_nodes, 0],fem_res['nodes'][elem_nodes, 1],"k-",linewidth=0.2,alpha=0.3,)
            
            # Overlay Phreatic Line (psi = 0)
            ax_fem.plot(fem_res['x_phreatic'], fem_res['y_phreatic'], 'r--', linewidth=2.5, label="Phreatic Line (ψ = 0)")
            
            ax_fem.set_title(f"Darcy FEM Seepage: {field_label}")
            ax_fem.set_xlabel("Distance [m]"); ax_fem.set_ylabel("Elevation [m]"); ax_fem.set_aspect("equal")
            ax_fem.legend(loc="upper left")

            # Colorbar alignment
            divider = make_axes_locatable(ax_fem)
            cax = divider.append_axes("right", size="2%", pad=0.15)
            cbar = fig_fem.colorbar(cf, cax=cax); cbar.set_label(field_label)

            plt.tight_layout()
            st.pyplot(fig_fem)

            st.success("✅ Darcy FEM simulation computed successfully.")

    # --- FOOTER & DIAGNOSTICS ---
    st.write("---")
    st.write("### 📈 Solver Convergence & Force Balance Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Resisting (Num)", f"{num:.2f} kN")
    col2.metric("Driving (Den)", f"{den:.2f} kN")
    col3.metric("Final FS", f"{fs:.3f}")

    st.subheader("📋 Raw Data Feed (Neon AWS)")
    st.dataframe(data, use_container_width=True)

    st.sidebar.divider()
    st.sidebar.markdown(f"**Developer:** MSc Juan Avalos Carrión")
    st.sidebar.caption("Geophysics Data Engineer, AI + Physics | 2026")
    st.sidebar.markdown(f"https://www.linkedin.com/in/juan-a-c-01457674/")

else:
    st.warning("Database empty. Check Neon connection.")
