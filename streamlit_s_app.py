import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine, text
from app.models_s import calculate_slope_stability # Using the new models_s
import plotly.graph_objects as go

# 1. DATABASE CONFIGURATION
DB_URI = st.secrets["NEON_DB_URI"] 
engine = create_engine(DB_URI, connect_args={"ssl_context": True})

st.set_page_config(page_title="Seismic Tailing Safety", layout="wide")

st.title("🛡️ Tailing Dam Safety System - Peru")
st.markdown("Integrated Bishop Stability Model with **Pseudo-static Seismic Analysis**. JUAN A.C. 2026")

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
                {'range': [0, 1.1], 'color': '#ff4b4b'}, # Red (Critical for Seismic)
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
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⏱️ Data Selection")
    selected_time = st.sidebar.selectbox("Select Timestamp", data['timestamp'])
    current_row = data[data['timestamp'] == selected_time].iloc[0]
    u_latest = current_row['pore_pressure']

    st.sidebar.header("🌋 Seismic Analysis")
    kh = st.sidebar.slider("Seismic Coeff (kh)", 0.0, 0.3, 0.15, step=0.01, 
                           help="Peruvian Standard E.050: 0.15 for Coast/High Risk")

    # --- TABS LAYOUT ---
    tab1, tab2 = st.tabs(["🎮 Manual Explorer", "🔥 Global Heatmap"])

    with tab1:
        st.sidebar.header("🔴 Slip Circle Geometry")
        xc = st.sidebar.slider("Center X (xc)", 20.0, 150.0, 75.0)
        yc = st.sidebar.slider("Center Y (yc)", 30.0, 150.0, 85.0)
        R = st.sidebar.slider("Radius (R)", 10.0, 100.0, 65.0)

        # Physics Run with kh
        fs, slices, water_line, history, num, den = calculate_slope_stability(xc, yc, R, u_latest, kh=kh)

        col1, col2 = st.columns([1, 3])
                
        with col1:
            if fs:
                st.plotly_chart(plot_fs_gauge(fs), use_container_width=True)
                if fs < 1.0: st.error("🚨 SEISMIC COLLAPSE")
                elif fs < 1.2: st.warning("⚠️ CRITICAL VULNERABILITY")
                # elif fs <=50: st.success("✅ SEISMICALLY STABLE")
                elif fs==0: st.error("❌ No Intersection found.")
                else: st.warning("⚖️ Equilibrium reached: The driving forces are too small to cause a slide for this specific circle.")
            else:
                st.error("No Intersection")
            st.write(f"**Fs:** {fs}")
            st.write(f"**Pore Pressure:** {u_latest} kPa")
            st.write(f"**Head:** {round(u_latest/9.81, 2)} m")

        with col2:
            from matplotlib.lines import Line2D # Necessary for the legend proxy
            
            fig, ax = plt.subplots(figsize=(10, 6))
            dx, dy = np.array([40, 70, 100, 130]), np.array([10, 45, 45, 14])
            ax.plot(dx, dy, 'k-', linewidth=3)
            ax.fill_between(dx, dy, color='navajowhite', alpha=0.8)
            ax.plot(water_line[0], water_line[1], 'b--', label="Phreatic Line")
            ax.scatter([80], [10], color='blue', s=100, zorder=5, label="PZ-01 Sensor")
            
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(xc + R*np.cos(theta), yc + R*np.sin(theta), 'r--', alpha=0.4)
            ax.scatter([xc], [yc], color='red', marker='+', s=100)
            # CREATE PROXY ARTIST FOR LEGEND (The trick for arrows)
            seismic_arrow_legend = Line2D([0], [0], color='red', marker='>', linestyle='-', 
                                          markersize=10, label=f'Seismic Force (kh={kh})')
            
            if slices:
                for s in slices:
                    ax.bar(s['x_mid'], s['h'], width=s['b'], bottom=s['y_bot'], 
                           color='orange', alpha=0.5, edgecolor='black', linewidth=0.2)
                    # SEISMIC VECTOR PHYSICS (NEW):
                    if kh > 0 and s['h'] > 0: # Only draw if there is soil and kh > 0
                        # Vector Origin: The vertical midpoint of the slice
                        y_midpoint = s['y_bot'] + (s['h'] / 2)
                        
                        # Magnitude of the push: proportional to slice height and kh
                        # (We scale it so it looks good on the plot)
                        vector_magnitude = - kh * s['h'] * 1.1 # 0.5 
                        
                        # Draw the red vector (Fseismic = W * kh)
                        # Pointing Right (Out of slope)
                        ax.arrow(s['x_mid'], y_midpoint, vector_magnitude, 0, 
                                 head_width=1.5, head_length=1.0, fc='red', ec='red', 
                                 alpha=0.8, zorder=10)
            # We manually collect the handles to include our proxy seismic patch
            handles, labels = ax.get_legend_handles_labels()
            if kh > 0:
                handles.append(seismic_arrow_legend)
                
            ax.set_ylim(0, 120); ax.set_xlim(20, 150); ax.set_aspect('equal')
            ax.legend(handles=handles, loc='upper left'); ax.grid(True, alpha=0.2)
            st.pyplot(fig)
            
    with tab2:
        st.subheader("Seismic Grid Search")
        st.write("Calculates FS for a grid of centers using the current Radius.")
        if st.button("🚀 Start Global Seismic Scan"):
            grid_x = np.linspace(30, 140, 15)
            grid_y = np.linspace(60, 140, 15)
            fs_matrix = np.zeros((len(grid_y), len(grid_x)))
            progress_text = "Analyzing slope stability surfaces..."
            my_bar = st.progress(0, text=progress_text)

            for i, py in enumerate(grid_y):
                for j, px in enumerate(grid_x):
                    val, _, _ = calculate_slope_stability(px, py, R, u_latest, kh=kh)
                    # Filter out artifacts/singularities
                    fs_matrix[i, j] = val if (val and 0 < val < 10) else np.nan
                my_bar.progress((i + 1) / len(grid_y))

            fig_h, ax_h = plt.subplots(figsize=(10, 8))
            sns.heatmap(fs_matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                        xticklabels=np.round(grid_x, 0), yticklabels=np.round(grid_y, 0), ax=ax_h, cbar_kws={'label': 'Factor of Safety'})
            ax_h.invert_yaxis()
            ax_h.set_title(f"Minimum Safety Zones for Radius {R}m")
            ax_h.set_xlabel("Center X (m)")
            ax_h.set_ylabel("Center Y (m)")
            st.pyplot(fig_h)
            
            min_found = np.nanmin(fs_matrix)
            st.info(f"The most critical center in this zone has an FS of: **{min_found}**")

    #with st.expander("📈 View Solver Convergence"):
    fig_conv, ax_conv = plt.subplots(figsize=(6, 2))
    ax_conv.plot(history, marker='o', linestyle='-', color='purple')
    ax_conv.set_title("Bishop Iteration Path")
    ax_conv.set_xlabel("Iteration Step")
    ax_conv.set_ylabel("Factor of Safety")
    ax_conv.grid(True, alpha=0.3)
    st.pyplot(fig_conv)
    st.write(f"Converged in **{len(history)-1}** steps.")

    st.write("### 📐 Slice Angle Distribution")
    # 1. Extract data from the list of dictionaries
    x_coords = [s['x_mid'] for s in slices]
    alphas = [np.degrees(s['alpha_rad']) for s in slices] # alpha_rad from your list
    # 2. Create the plot
    fig_alpha, ax_alpha = plt.subplots(figsize=(8, 4))
    ax_alpha.plot(x_coords, alphas, marker='o', color='teal', label='Base Angle (α)')
    ax_alpha.axhline(0, color='black', linestyle='--', alpha=0.5)
    # Labels and Styling
    ax_alpha.set_xlabel("X-Coordinate of Slice (m)"); ax_alpha.set_ylabel("Angle α (degrees)")
    ax_alpha.set_title("Distribution of Slip Surface Inclination")
    ax_alpha.grid(True, linestyle=':', alpha=0.6)
    # Fill the area to show Driving vs Resisting zones
    ax_alpha.fill_between(x_coords, alphas, 0, where=(np.array(alphas) > 0), 
                          color='salmon', alpha=0.5, label='Driving Zone')
    ax_alpha.fill_between(x_coords, alphas, 0, where=(np.array(alphas) < 0), 
                          color='skyblue', alpha=0.5, label='Resisting Zone')
    
    ax_alpha.legend()
    st.pyplot(fig_alpha) # !!

    st.write("### ⚖️ Force Balance Analysis") # !!!
    # Display as Metrics for quick reading
    col1, col2, col3 = st.columns(3)
    col1.metric("Resisting (Num)", f"{num:.2f} kN")
    col2.metric("Driving (Den)", f"{den:.2f} kN")
    col3.metric("Final FS", f"{fs:.3f}")
    # 2. Create a Comparison Bar Chart
    #import pandas as pd
    #force_data = pd.DataFrame({
    #    "Force Type": ["Resisting (Strength)", "Driving (Load)"],
    #    "Value [kN]": [num, den]
    #})
    # Use st.bar_chart or Plotly for a more professional look
    #st.bar_chart(data=force_data, x="Force Type", y="Value [kN]", color="#2e7d32" if fs > 1.5 else "#d32f2f")
    #st.caption("The Factor of Safety is simply the Green bar divided by the Red bar.") # !!
    
    st.write("---")
    st.subheader("📋 Raw Data Feed (Neon AWS)")
    st.dataframe(data, use_container_width=True)

else:
    st.warning("Database empty. Check Neon connection.")
