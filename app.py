import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import yaml
import random
import streamlit.components.v1 as components
from omegaconf import OmegaConf, DictConfig
from shapely.geometry import shape
import plotly.express as px

# Import the team's new modular architecture
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.core.types import WalkerParameters, OrbitType
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.traffic.profiles import generate_users
from hybrid_ntn_optimizer.terrestrial.coverage import generate_terrestrial_network
from hybrid_ntn_optimizer.simulation.full_pipeline import run_daily_mobility_simulation
from hybrid_ntn_optimizer.visualization.plots import build_h3_geojson, build_bs_coverage_geojson

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="5G Network Digital Twin", layout="wide")
st.title("Hybrid TN and LEO Network Digital Twin")
st.markdown("A spatio-temporal simulation engine for 5G heterogeneous networks.")

# ==========================================
# SIDEBAR: USER INPUTS (LIVE CONTROLS)
# ==========================================
st.sidebar.header("Simulation Parameters")

st.sidebar.subheader("1. Population Settings")
TOTAL_USERS = st.sidebar.slider("Total Simulated Users", min_value=0, max_value=5000, value=1000, step=500)
CITY_RATIO = st.sidebar.slider("City vs Rural Ratio", min_value=0.1, max_value=0.9, value=0.7, step=0.1)

st.sidebar.subheader("2. Infrastructure Settings")
TN_POP_THRESHOLD = st.sidebar.slider("Tower Threshold (Min Users)", min_value=10, max_value=100, value=50, step=5)
USERS_PER_CLUSTER = st.sidebar.slider("Target Users per Tower (K-Means)", min_value=5, max_value=100, value=20, step=5)
TN_BS_CAPACITY_GBPS = st.sidebar.slider("TN Tower Capacity (Gbps)", min_value=1, max_value=50, value=10, step=1)
TN_BS_CAPACITY_MBPS = TN_BS_CAPACITY_GBPS * 1000
TN_BW_MHZ = st.sidebar.slider("BS Bandwidth (MHz)", min_value=10, max_value=400, value=100, step=10)

# --- Advanced 5G RF Parameters Dropdown ---
with st.sidebar.expander("Advanced 5G RF Parameters"):
    TN_COVERAGE_RADIUS = st.slider("Coverage Radius (km)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    TN_P_TX = st.slider("BS Transmit Power (dBm)", min_value=20.0, max_value=60.0, value=43.0, step=1.0)
    TN_G_TX = st.slider("BS Antenna Gain (dBi)", min_value=0.0, max_value=30.0, value=15.0, step=1.0)
    TN_G_RX = st.slider("UE Receive Gain (dBi)", min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
    # We display GHz/MHz for the user, but will multiply by 1e9/1e6 for the backend code
    TN_FREQ_GHZ = st.slider("Carrier Frequency (GHz)", min_value=0.5, max_value=6.0, value=3.5, step=0.1)
    TN_SINR_MIN = st.slider("Min SINR (dB)", min_value=-10.0, max_value=10.0, value=-3.0, step=0.5)
    TN_SHADOWING = st.slider("Shadowing Std Dev (dB)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
    TN_BODY_LOSS = st.slider("Body/Penetration Loss (dB)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)

st.sidebar.subheader("3. LEO Constellation Settings")
SAT_ALTITUDE = st.sidebar.slider("Satellite Altitude (km)", min_value=300.0, max_value=1500.0, value=550.0, step=50.0)
TOTAL_SATS = st.sidebar.select_slider("Total Satellites in Constellation", options=[72, 324, 648, 1584, 4000], value=1584)
NTN_BW_MHZ = st.sidebar.slider("NTN Beam Bandwidth (MHz)", min_value=10, max_value=100, value=40, step=10)
SAT_EIRP = st.sidebar.slider("Satellite EIRP (dBW)", min_value=20.0, max_value=60.0, value=50.0, step=1.0)

st.sidebar.subheader("4. Diurnal Traffic Model")
EVENING_PEAK_HOUR = st.sidebar.slider("Evening Peak Time (Hour)", min_value=16.0, max_value=23.0, value=20.0, step=0.5)

st.sidebar.subheader("5. User Traffic Profiles")
st.sidebar.caption("Check the boxes to include these 3GPP usage scenarios.")
use_light = st.sidebar.checkbox("Light Users (mMTC | 0.1 - 1 Mbps)", value=True)
use_medium = st.sidebar.checkbox("Medium Users (Nominal | 1.5 - 5 Mbps)", value=True)
use_heavy = st.sidebar.checkbox("Heavy Users (eMBB | 10 - 25 Mbps)", value=True)

active_count = sum([use_light, use_medium, use_heavy])
if active_count == 0:
    st.sidebar.error("Please select at least one user profile to run the simulation.")
    st.stop()
prob_share = 1.0 / active_count

st.sidebar.subheader("6. Simulation Engine")
SIM_DURATION = st.sidebar.slider("Simulation Duration (Seconds)", min_value=3600, max_value=86400, value=18000, step=3600)
TIME_STEP = st.sidebar.slider("Time Step (Seconds)", min_value=600, max_value=3600, value=3600, step=600)

st.sidebar.markdown("---")
st.sidebar.info("Note: The dashboard is live. Adjusting any parameter will update the backend simulation automatically.")

# ==========================================
# MAP HELPERS & CUSTOM RENDERER
# ==========================================
with open(r"E:\berkay\NTN\configs\scenario\ontario_full.yaml", "r", encoding="utf-8") as f:
    ontario_yaml = yaml.safe_load(f)

ONTARIO_GEOM = shape(ontario_yaml["geojson_geometry"])
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = ONTARIO_GEOM.bounds
center_lat, center_lon = (LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2
city_centers = [(43.65, -79.38), (45.42, -75.69), (43.25, -79.87), (42.98, -81.25), (44.23, -76.49), (46.49, -81.01)]
city_names = ['Toronto', 'Ottawa', 'Hamilton', 'London', 'Kingston', 'Sudbury']

def get_boundary_coords(geom):
    x_all, y_all = [], []
    polygons = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polygons:
        x, y = poly.exterior.xy
        x_all.extend(list(x) + [None])
        y_all.extend(list(y) + [None])
    return x_all, y_all


def render_custom_dashboard_animation(region, users, base_stations, beam_data, user_data, duration_s, time_step_s, filename):
    hex_geojson = build_h3_geojson(region.cells)
    bs_coverage_geojson = build_bs_coverage_geojson(base_stations)
    time_steps = list(range(0, duration_s + time_step_s, time_step_s))
    all_h3_ids = [cell.h3_id for cell in region.cells]
    
    fig = go.Figure()

    initial_beams = [b["h3_id"] for b in beam_data if b["time_s"] == 0]
    initial_z = [1 if h3_id in initial_beams else 0 for h3_id in all_h3_ids]
    
    # Restored to Choroplethmapbox
    fig.add_trace(go.Choroplethmapbox(
        geojson=hex_geojson, locations=all_h3_ids, z=initial_z,
        colorscale=[[0, "rgba(50, 50, 50, 0.1)"], [1, "rgba(0, 255, 100, 0.4)"]], 
        zmin=0, zmax=1, marker_opacity=0.6, marker_line_width=1, showscale=False,
        name="Satellite Beams", hoverinfo="skip"
    ))

    user_states = [("TN", "deepskyblue", "TN Served (5G)"), ("LEO", "hotpink", "NTN Served (Satellite)"),
                   ("DROPPED", "red", "Dropped (Outage)"), ("IDLE", "gray", "Idle")]
    initial_users = [u for u in user_data if u["Hour"] == "Hour 0.0"]
    
    for state_id, color, label in user_states:
        state_users = [u for u in initial_users if u["State"] == state_id]
        # Restored to Scattermapbox
        fig.add_trace(go.Scattermapbox(
            lat=[u["Lat"] for u in state_users], lon=[u["Lon"] for u in state_users],
            mode='markers', marker=dict(size=6, color=color, opacity=0.9),
            name=label, hoverinfo='text',
            text=[f"User {u['User_ID']}<br>State: {state_id}" for u in state_users]
        ))

    # Restored to Scattermapbox
    fig.add_trace(go.Scattermapbox(
        lat=[bs.lat for bs in base_stations], lon=[bs.lon for bs in base_stations],
        mode='markers', marker=dict(size=10, color='orange', symbol='circle'),
        name='5G Base Stations', hoverinfo='text',
        text=[f"Tower {bs.bs_id}<br>Radius: {bs.coverage_radius_km:.2f} km" for bs in base_stations]
    ))

    frames = []
    slider_steps = []
    for t_s in time_steps:
        hour_str = f"Hour {t_s / 3600.0:.1f}"
        active_beams = [b["h3_id"] for b in beam_data if b["time_s"] == t_s]
        frame_z = [1 if h3_id in active_beams else 0 for h3_id in all_h3_ids]
        
        frame_data = [go.Choroplethmapbox(z=frame_z)]
        
        frame_users = [u for u in user_data if u["Hour"] == hour_str]
        for state_id, _, _ in user_states:
            state_users = [u for u in frame_users if u["State"] == state_id]
            frame_data.append(go.Scattermapbox(lat=[u["Lat"] for u in state_users], lon=[u["Lon"] for u in state_users], text=[f"User {u['User_ID']}<br>State: {state_id}" for u in state_users]))
            
        frames.append(go.Frame(name=hour_str, data=frame_data, traces=[0, 1, 2, 3, 4]))
        slider_steps.append({"args": [[hour_str], {"frame": {"duration": 800, "redraw": True}, "mode": "immediate"}], "label": hour_str, "method": "animate"})

    fig.frames = frames

    mapbox_layers = [dict(source=bs_coverage_geojson, type="fill", color="rgba(255, 165, 0, 0.25)")]
    raw_geometry = OmegaConf.to_container(region.geojson_geometry, resolve=True) if isinstance(region.geojson_geometry, DictConfig) else region.geojson_geometry
    mapbox_layers.append(dict(source=raw_geometry, type="line", color="cyan", line=dict(width=2)))

    fig.update_layout(
        title="Hybrid NTN-TN Real-Time Traffic Routing (System Level Engine)",
        template="plotly_dark", height=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        mapbox=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=4.5, layers=mapbox_layers),
        margin={"r":0,"t":50,"l":0,"b":0},
        updatemenus=[{"buttons": [{"args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}], "label": "Play ▶", "method": "animate"},
                                  {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "Pause ⏸", "method": "animate"}],
                      "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"}],
        sliders=[{"active": 0, "yanchor": "top", "xanchor": "left", "currentvalue": {"font": {"size": 20}, "prefix": "Time: "}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": slider_steps}]
    )
    fig.write_html(filename)

# ==========================================
# DYNAMIC CONFIGURATION BUILDER
# ==========================================
cfg = OmegaConf.create({
    "random_seed": 42,
    "epoch_utc": "2024-01-01T00:00:00",
    "scenario": {
        "name": "Ontario_Province",
        "h3_resolution": ontario_yaml.get("h3_resolution", 3),
        "geojson_geometry": ontario_yaml["geojson_geometry"]
    },
    "constellation": {
        "name": "Live-Simulation-Shell",
        "total_satellites": TOTAL_SATS,
        "num_planes": 72 if TOTAL_SATS == 1584 else max(1, int(TOTAL_SATS/18)), 
        "phasing": 1,
        "inclination_deg": 53.0,
        "altitude_km": SAT_ALTITUDE,
        "eirp_dbw": SAT_EIRP,
        "g_t_db": -15.5,
        "min_elevation_deg": 25.0,
        "apply_j2": True,
        "max_spot_beams": 32,
        "beam_radius_nadir_km": 120.0,
        "max_steering_angle_deg": 45.0,
        "freq_ghz": 2.2, 
        "bandwidth_hz": NTN_BW_MHZ * 1000000, 
        "sinr_min_db": 0.0,
        "theta_3db_deg": 2.5,
        "sll_db": 25.0,
        "weather_loss_db": 1.0
    },
    "population": {
        "total_city_users": int(TOTAL_USERS * CITY_RATIO),
        "total_rural_users": TOTAL_USERS - int(TOTAL_USERS * CITY_RATIO),
        "city_scatter_std_dev": 0.15,
        "cities": {
            "Toronto": {"coords": [43.65, -79.38], "weight": 0.70},
            "Ottawa": {"coords": [45.42, -75.69], "weight": 0.11},
            "Hamilton": {"coords": [43.25, -79.87], "weight": 0.09},
            "London": {"coords": [42.98, -81.25], "weight": 0.06},
            "Kingston": {"coords": [44.23, -76.49], "weight": 0.02},
            "Sudbury": {"coords": [46.49, -81.01], "weight": 0.02}
        },
        "mobility": {
            "num_attractors": 3,
            "zipf_alpha": 1.2,
            "pareto_beta": 1.75,
            "delta_r0_km": 1.5,
            "cutoff_kappa_km": 80.0,
            "night_hours_start": 22,
            "night_hours_end": 6,
            "night_move_chance": 0.1,
            "day_move_chance": 0.4,
            "gps_wander_std_dev": 0.005
        },
        "traffic": {
            "diurnal_curve": {
                "base_traffic_multiplier": 0.2,
                "noon_peak": {"center_hour": 12.0, "width_hours": 3.0, "height_multiplier": 0.5},
                "evening_peak": {"center_hour": EVENING_PEAK_HOUR, "width_hours": 2.5, "height_multiplier": 1.0}
            },
            "profiles": {} 
        }
    },
"terrestrial": {
        "density_threshold": TN_POP_THRESHOLD,
        "bs_capacity_mbps": float(TN_BS_CAPACITY_MBPS),
        "users_per_cluster_ratio": USERS_PER_CLUSTER,
        "p_tx_dbm": TN_P_TX,
        "g_tx_dbi": TN_G_TX,
        "g_rx_ue_dbi": TN_G_RX,
        "carrier_freq_hz": TN_FREQ_GHZ * 1e9,   # Converts GHz slider to Hz
        "bandwidth_hz": TN_BW_MHZ * 1e6,        # Converts MHz slider to Hz
        "sinr_min_db": TN_SINR_MIN,
        "shadowing_std_dev_db": TN_SHADOWING,         
        "body_loss_db": TN_BODY_LOSS,
        "use_physical_radius": True,
        "fixed_coverage_radius_km": True,    
        "coverage_radius_km": TN_COVERAGE_RADIUS           
    },
    "simulation": {
        "duration_s": SIM_DURATION,
        "time_step_s": TIME_STEP,
    }
})

if use_light: cfg.population.traffic.profiles.light = {"probability": prob_share, "min_mbps": 0.1, "max_mbps": 1.0}
if use_medium: cfg.population.traffic.profiles.medium = {"probability": prob_share, "min_mbps": 1.5, "max_mbps": 5.0}
if use_heavy: cfg.population.traffic.profiles.heavy = {"probability": prob_share, "min_mbps": 10.0, "max_mbps": 25.0}

# ==========================================
# SIMULATION EXECUTION
# ==========================================
with st.spinner("Executing Geographic Tessellation..."):
    active_region = Region(name=cfg.scenario.name, geojson_geometry=cfg.scenario.geojson_geometry, h3_resolution=cfg.scenario.h3_resolution)
    tessellate_region(active_region, pad_edges=True)

with st.spinner("Deploying LEO Satellite Constellation & SGP4 Propagators..."):
    walker_params = WalkerParameters(
        total_satellites=cfg.constellation.total_satellites, num_planes=cfg.constellation.num_planes,
        phasing=cfg.constellation.phasing, inclination_deg=cfg.constellation.inclination_deg,
        altitude_km=cfg.constellation.altitude_km, orbit_type=OrbitType.LEO
    )
    leo = LEOConstellation(
        params=walker_params, name=cfg.constellation.name, eirp_dbw=cfg.constellation.eirp_dbw,
        g_t_db=cfg.constellation.g_t_db, max_spot_beams=cfg.constellation.max_spot_beams,
        beam_radius_nadir_km=cfg.constellation.beam_radius_nadir_km, max_steering_angle_deg=cfg.constellation.max_steering_angle_deg
    )

with st.spinner("Spawning Heterogeneous User Population..."):
    users = generate_users(cfg, active_region)

with st.spinner("Executing K-Means for Terrestrial Infrastructure..."):
    towers = generate_terrestrial_network(cfg, users, active_region.h3_resolution)

with st.spinner("Running Master RF Simulation Loop (Link Budgets & Admission Control)..."):
    beam_animation_data, user_animation_data = run_daily_mobility_simulation(
        cfg=cfg, users=users, base_stations=towers, leo=leo, region=active_region
    )

with st.spinner("Compiling Final HTML Visualizations..."):
    html_filename = "Final_Animation.html"
    render_custom_dashboard_animation(
        region=active_region, users=users, base_stations=towers, 
        beam_data=beam_animation_data, user_data=user_animation_data,
        duration_s=cfg.simulation.duration_s, time_step_s=cfg.simulation.time_step_s, filename=html_filename
    )

st.success("Simulation Complete. Results are rendered below.")

# ==========================================
# DISPLAY DASHBOARD
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Interactive Simulation Map", "STEPS Mobility", "Traffic Analytics", "Network Utilization", "Data Exports"])

df_users_anim = pd.read_csv("user_hourly_states.csv") if os.path.exists("user_hourly_states.csv") else pd.DataFrame()
df_summary = pd.read_csv("system_summary_table.csv") if os.path.exists("system_summary_table.csv") else pd.DataFrame()
df_usage = pd.read_csv("network_usage_data.csv") if os.path.exists("network_usage_data.csv") else pd.DataFrame()

with tab1:
    st.subheader("Hybrid Network Traffic Routing Animation")
    st.markdown("This map dynamically renders the output of the full physics pipeline, including 5G Admission Control and LEO Spot Beam steering.")
    
    if os.path.exists(html_filename):
        with open(html_filename, 'r', encoding='utf-8') as f:
            html_content = f.read()
        # FIX: Use components.html and increase height to 850 so the slider fits perfectly!
        components.html(html_content, height=850)
    else:
        st.error("Visualization file was not generated.")

with tab2:
    if not df_users_anim.empty:
        bx, by = get_boundary_coords(ONTARIO_GEOM)
        colA, colB = st.columns(2)
        
        with colA:
            fig2A = go.Figure()
            fig2A.add_trace(go.Scattermap(lat=by, lon=bx, mode='lines', line=dict(color='white', width=1.0), showlegend=False, hoverinfo='skip'))
            unique_users = df_users_anim['User_ID'].unique()
            tracked_users = random.sample(list(unique_users), min(8, len(unique_users)))
            plot_colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'lightgreen']
            
            for idx, uid in enumerate(tracked_users):
                user_path = df_users_anim[df_users_anim['User_ID'] == uid]
                fig2A.add_trace(go.Scattermap(lat=user_path['Lat'], lon=user_path['Lon'], mode='lines+markers', line=dict(color=plot_colors[idx]), marker=dict(size=5), name=f"User {uid}"))
            fig2A.update_layout(title="User Trajectories Over Time", map=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=4.2), height=550, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2A, width="stretch")

        with colB:
            fig2B = go.Figure()
            fig2B.add_trace(go.Scattermap(lat=by, lon=bx, mode='lines', line=dict(color='white', width=1.0), showlegend=False, hoverinfo='skip'))
            fig2B.add_trace(go.Scattermap(lat=df_users_anim['Lat'], lon=df_users_anim['Lon'], mode='markers', marker=dict(color='cyan', size=3, opacity=0.03), showlegend=False))
            fig2B.update_layout(title="Spatial Attractor Density", map=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=4.2), height=550, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2B, width="stretch")

with tab3:
    if not df_summary.empty:
        fig3 = go.Figure()
        df_summary['Continuous_Hour'] = df_summary['Time_s'] / 3600.0
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Total_Demand_Mbps'], mode='lines', name='Total Demand (Mbps)', line=dict(color='cyan', width=3)))
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Served_TN_Mbps'], mode='lines', name='Served by TN (5G)', line=dict(color='deepskyblue', width=2, dash='dash')))
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Served_NTN_Mbps'], mode='lines', name='Served by LEO (Sat)', line=dict(color='magenta', width=2, dash='dot')))
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Dropped_Traffic_Mbps'], mode='lines', name='Dropped Traffic (Outage)', line=dict(color='red', width=2)))
        
        fig3.update_layout(xaxis_title="Time of Day (Hours)", yaxis_title="Data Load (Mbps)", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig3.update_xaxes(tickvals=list(range(0, 25, 2)), gridcolor='rgba(0, 0, 0, 0.1)')
        fig3.update_yaxes(gridcolor='rgba(0, 0, 0, 0.1)')
        st.plotly_chart(fig3, width="stretch", theme=None)

with tab4:
    if not df_usage.empty:
        st.subheader("Physical Hardware Utilization")
        st.markdown("Tracks the exhaustion of physical radio spectrum (MHz) across 5G Towers and LEO Satellites.")
        
        # Calculate average utilization for each hour
        usage_summary = df_usage.groupby(['Hour', 'Network_Type'])['Utilization_%'].mean().reset_index()
        # Convert hour strings back to continuous float for plotting
        usage_summary['Time'] = usage_summary['Hour'].str.extract('(\d+\.\d+)').astype(float)
        
        # FIX: Sort by Network Type first, then by Time, so the lines never cross backwards!
        usage_summary = usage_summary.sort_values(by=["Network_Type", "Time"])
        
        fig4 = px.line(usage_summary, x="Time", y="Utilization_%", color="Network_Type", 
                       title="Average Spectrum Utilization (%)",
                       color_discrete_map={"5G_TN": "orange", "LEO_NTN": "magenta"})
                       
        fig4.update_layout(xaxis_title="Time of Day (Hours)", yaxis_title="Bandwidth Used (%)", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig4.update_yaxes(range=[0, 105], gridcolor='rgba(0, 0, 0, 0.1)')
        fig4.update_xaxes(tickvals=list(range(0, 25, 2)), gridcolor='rgba(0, 0, 0, 0.1)')
        st.plotly_chart(fig4, width="stretch", theme=None)

with tab5:
    st.subheader("Data Exports & Previews")
    st.markdown("Preview the raw data generated by the simulation engine and download the CSVs.")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### User Data")
        if os.path.exists("users_initial_state.csv"):
            df_init = pd.read_csv("users_initial_state.csv")
            st.dataframe(df_init, height=250, width="stretch")
            with open("users_initial_state.csv", "rb") as f:
                st.download_button("Download users_initial_state.csv", data=f, file_name="users_initial_state.csv", mime="text/csv")
        
        st.write("") # Add a little space
        
        if os.path.exists("user_hourly_states.csv"):
            df_hourly = pd.read_csv("user_hourly_states.csv")
            st.dataframe(df_hourly, height=250, width="stretch")
            with open("user_hourly_states.csv", "rb") as f:
                st.download_button("Download user_hourly_states.csv", data=f, file_name="user_hourly_states.csv", mime="text/csv")
                
    with c2:
        st.markdown("#### System & Usage Data")
        if os.path.exists("system_summary_table.csv"):
            df_sys = pd.read_csv("system_summary_table.csv")
            st.dataframe(df_sys, height=250, width="stretch")
            with open("system_summary_table.csv", "rb") as f:
                st.download_button("Download system_summary.csv", data=f, file_name="system_summary_table.csv", mime="text/csv")
        
        st.write("")
        
        if os.path.exists("network_usage_data.csv"):
            df_net = pd.read_csv("network_usage_data.csv")
            st.dataframe(df_net, height=250, width="stretch")
            with open("network_usage_data.csv", "rb") as f:
                st.download_button("Download network_usage_data.csv", data=f, file_name="network_usage_data.csv", mime="text/csv")
                
    with c3:
        st.markdown("#### Diagnostics Log")
        if os.path.exists("detailed_drop_log.csv"):
            df_drop = pd.read_csv("detailed_drop_log.csv")
            st.dataframe(df_drop, height=250, width="stretch")
            with open("detailed_drop_log.csv", "rb") as f:
                st.download_button("Download detailed_drop_log.csv", data=f, file_name="detailed_drop_log.csv", mime="text/csv")