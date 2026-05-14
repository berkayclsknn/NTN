import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import random
import yaml
from shapely.geometry import shape, Point
from shapely.prepared import prep

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
TOTAL_USERS = st.sidebar.slider("Total Simulated Users", min_value=500, max_value=5000, value=1000, step=500)
CITY_RATIO = st.sidebar.slider("City vs Rural Ratio", min_value=0.1, max_value=0.9, value=0.7, step=0.1)

num_city_users = int(TOTAL_USERS * CITY_RATIO)
num_rural_users = TOTAL_USERS - num_city_users

st.sidebar.subheader("2. Infrastructure Settings")
TN_POP_THRESHOLD = st.sidebar.slider("Tower Threshold (Min Users)", min_value=10, max_value=100, value=50, step=5)
TN_BS_CAPACITY_GBPS = st.sidebar.slider("TN Tower Capacity (Gbps)", min_value=1, max_value=50, value=10, step=1)
TN_BS_CAPACITY_MBPS = TN_BS_CAPACITY_GBPS * 1000

st.sidebar.subheader("3. Diurnal Traffic Model")
EVENING_PEAK_HOUR = st.sidebar.slider(
    "Evening Peak Time (Hour)", 
    min_value=16.0, 
    max_value=23.0, 
    value=20.0, 
    step=0.5,
    help="Shifts the center of the Gaussian evening traffic peak (e.g., 20.0 is 8:00 PM)."
)

st.sidebar.subheader("4. User Traffic Profiles")
st.sidebar.caption("Check the boxes to include these 3GPP usage scenarios.")
use_light = st.sidebar.checkbox("Light Users (mMTC | 0.1 - 1 Mbps)", value=True)
use_medium = st.sidebar.checkbox("Medium Users (Nominal | 1.5 - 5 Mbps)", value=True)
use_heavy = st.sidebar.checkbox("Heavy Users (eMBB | 10 - 25 Mbps)", value=True)

active_profiles = []
if use_light: active_profiles.append("Light")
if use_medium: active_profiles.append("Medium")
if use_heavy: active_profiles.append("Heavy")

if len(active_profiles) == 0:
    st.sidebar.error("Please select at least one user profile to run the simulation.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info("Note: The dashboard is live. Adjusting any parameter will update the simulation automatically.")

# ==========================================
# DATA & CLASSES
# ==========================================
@st.cache_resource 
def load_ontario_map():
    with open("E:\\berkay\\NTN\\ontario_full.yaml", "r", encoding="utf-8") as f:
        ontario_yaml = yaml.safe_load(f)
    geom = shape(ontario_yaml["geojson_geometry"])
    return prep(geom), geom, geom.bounds, ontario_yaml.get("h3_resolution", 2)

ONTARIO_PREPARED, ONTARIO_GEOM, BOUNDS, H3_RESOLUTION = load_ontario_map()
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = BOUNDS

def random_point_inside_ontario():
    while True:
        lon = np.random.uniform(LON_MIN, LON_MAX)
        lat = np.random.uniform(LAT_MIN, LAT_MAX)
        if ONTARIO_PREPARED.contains(Point(lon, lat)):
            return lat, lon

def get_boundary_coords(geom):
    x_all, y_all = [], []
    polygons = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polygons:
        x, y = poly.exterior.xy
        x_all.extend(list(x) + [None])
        y_all.extend(list(y) + [None])
    return x_all, y_all

def get_hex_grid_coords(hex_radius):
    mean_lat = (LAT_MIN + LAT_MAX) / 2.0
    aspect_ratio = 1.0 / np.cos(np.radians(mean_lat))
    
    horiz_spacing = np.sqrt(3) * hex_radius * aspect_ratio
    vert_spacing = 1.5 * hex_radius
    
    rows = int((LAT_MAX - LAT_MIN) / vert_spacing) + 2
    cols = int((LON_MAX - LON_MIN) / horiz_spacing) + 2
    
    hex_x, hex_y = [], []
    angles = np.linspace(0, 2 * np.pi, 7) + (np.pi / 6)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    
    for row in range(rows):
        for col in range(cols):
            y = LAT_MIN + row * vert_spacing
            x = LON_MIN + col * horiz_spacing
            if row % 2 == 1: 
                x += horiz_spacing / 2
            
            if not ONTARIO_PREPARED.contains(Point(x, y)):
                continue
                
            x_verts = x + (hex_radius * aspect_ratio) * cos_a
            y_verts = y + hex_radius * sin_a
            hex_x.extend(x_verts.tolist() + [None])
            hex_y.extend(y_verts.tolist() + [None])
    return hex_x, hex_y

class User:
    def __init__(self, user_id, lat, lon, active_profiles):
        self.user_id = user_id
        
        num_attractors = 3
        ranks = np.arange(1, num_attractors + 1)
        raw_probs = 1.0 / (ranks ** 1.2)
        self.attractor_probs = raw_probs / np.sum(raw_probs)
        
        beta, delta_r0, kappa = 1.75, 1.5, 80.0 
        self.home_lat, self.home_lon = lat, lon
        self.attractors = [(self.home_lat, self.home_lon)]
        
        for _ in range(num_attractors - 1):
            accepted = False
            r_km = 0.0
            while not accepted:
                r_km = np.random.pareto(beta - 1.0) * delta_r0
                if np.random.rand() < np.exp(-r_km / kappa):
                    accepted = True
            r_deg = r_km / 111.0
            theta = np.random.uniform(0, 2 * np.pi)
            self.attractors.append((self.home_lat + (r_deg * np.sin(theta)), self.home_lon + (r_deg * np.cos(theta))))
            
        self.lat, self.lon = self.home_lat, self.home_lon
        self.history_lat, self.history_lon = [self.lat], [self.lon]        
        
        profile_roll = np.random.choice(active_profiles)
        if profile_roll == "Light": 
            self.user_type, self.base_demand_mbps = "Light (Text/Web)", np.random.uniform(0.1, 1.0)
        elif profile_roll == "Medium":
            self.user_type, self.base_demand_mbps = "Medium (Social/Video)", np.random.uniform(1.5, 5.0)
        elif profile_roll == "Heavy":
            self.user_type, self.base_demand_mbps = "Heavy (Gaming/4K)", np.random.uniform(10.0, 25.0)
            
        self.coverage_type, self.tn_cell_id = None, None     
        
    def get_demand_at_time(self, hour, evening_peak_hr):
        base_traffic = 0.2  
        noon_peak = 0.5 * np.exp(-((hour - 12.0)**2) / (2 * (3.0**2)))
        evening_peak = 1.0 * np.exp(-((hour - evening_peak_hr)**2) / (2 * (2.5**2)))
        diurnal_multiplier = base_traffic + noon_peak + evening_peak
        return self.base_demand_mbps * diurnal_multiplier

    def update_location_steps(self, hour):
        move_chance = 0.1 if (hour < 6 or hour > 22) else 0.4
        if np.random.rand() < move_chance:
            chosen_idx = np.random.choice(len(self.attractors), p=self.attractor_probs)
            target_lat, target_lon = self.attractors[chosen_idx]
            self.lat = target_lat + np.random.normal(0, 0.005)
            self.lon = target_lon + np.random.normal(0, 0.005)
        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)

# ==========================================
# MAIN EXECUTION
# ==========================================
np.random.seed(42)
users = []
city_centers = [(43.65, -79.38), (45.42, -75.69), (43.25, -79.87), (42.98, -81.25), (44.23, -76.49), (46.49, -81.01)]
city_names = ['Toronto', 'Ottawa', 'Hamilton', 'London', 'Kingston', 'Sudbury']

# Toronto (70%), Ottawa (11%), Hamilton (9%), London (6%), Kingston (2%), Sudbury (2%)
city_weights = [0.70, 0.11, 0.09, 0.06, 0.02, 0.02]

with st.spinner("Generating Population and Running STEPS Mobility..."):
    for i in range(num_city_users):
        center_idx = np.random.choice(len(city_centers), p=city_weights)
        center = city_centers[center_idx]
        users.append(User(user_id=i, lat=np.random.normal(center[0], 0.15), lon=np.random.normal(center[1], 0.15), active_profiles=active_profiles))

    for i in range(num_city_users, TOTAL_USERS):
        lat, lon = random_point_inside_ontario()
        users.append(User(user_id=i, lat=lat, lon=lon, active_profiles=active_profiles))

    coordinates = np.array([[u.lat, u.lon] for u in users])

with st.spinner("Clustering TN Towers..."):
    num_clusters = int(TOTAL_USERS / 40)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    potential_tn_locations = kmeans.cluster_centers_
    cluster_counts = pd.Series(cluster_labels).value_counts()
    valid_tn_towers = []

    for i, user in enumerate(users):
        cluster_id = cluster_labels[i]
        if cluster_counts[cluster_id] > TN_POP_THRESHOLD:
            user.coverage_type, user.tn_cell_id = "TN", cluster_id
        else:
            user.coverage_type, user.tn_cell_id = "LEO", "N/A (LEO)"

    for cluster_id, count in cluster_counts.items():
        if count > TN_POP_THRESHOLD:
            valid_tn_towers.append(potential_tn_locations[cluster_id])
    valid_tn_towers = np.array(valid_tn_towers)

with st.spinner("Simulating Traffic Data..."):
    user_data_export = []
    for u in users:
        row_data = {'User_ID': u.user_id, 'Home_TN_Cell': u.tn_cell_id, 'User_Profile': u.user_type, 'Base_Demand_Mbps': round(u.base_demand_mbps, 2)}
        for hour in range(24):
            row_data[f"Demand_Hour_{hour}"] = round(u.get_demand_at_time(hour, EVENING_PEAK_HOUR), 2)
            u.update_location_steps(hour)
            row_data[f"Lat_Hour_{hour}"], row_data[f"Lon_Hour_{hour}"] = round(u.lat, 4), round(u.lon, 4)
        user_data_export.append(row_data)

    df_users = pd.DataFrame(user_data_export)
    df_users['Home_TN_Cell'] = df_users['Home_TN_Cell'].astype(str) 

    LEO_TOTAL_CAPACITY_MBPS = 15000
    SCALE_FACTOR = 500
    time_series = pd.date_range(start="2025-01-01 00:00", end="2025-01-01 23:30", freq="30min")
    summary_data = []

    for ts in time_series:
        current_hour = ts.hour + (ts.minute / 60.0) 
        total_demand = total_served_tn = total_served_ntn = 0.0
        tn_tower_loads = {tower_id: 0.0 for tower_id in range(num_clusters)}
        leo_total_load = 0.0
        
        for u in users:
            user_demand = u.get_demand_at_time(current_hour, EVENING_PEAK_HOUR)
            total_demand += user_demand
            if u.coverage_type == "TN" and u.tn_cell_id != "N/A (LEO)":
                tn_tower_loads[u.tn_cell_id] += user_demand
            else:
                leo_total_load += user_demand
                
        for tower_id, load in tn_tower_loads.items():
            if load <= TN_BS_CAPACITY_MBPS:
                total_served_tn += load
            else:
                total_served_tn += TN_BS_CAPACITY_MBPS
                leo_total_load += load - TN_BS_CAPACITY_MBPS

        current_leo_cap = LEO_TOTAL_CAPACITY_MBPS * np.random.uniform(0.98, 1.02)
        if leo_total_load <= current_leo_cap:
            total_served_ntn += leo_total_load
        else:
            total_served_ntn += current_leo_cap

        summary_data.append({
            "Time Step": ts.strftime("%Y-%m-%d %H:%M"),
            "demand_mbps": round(total_demand * SCALE_FACTOR, 5),
            "served_tn_mbps": round(total_served_tn * SCALE_FACTOR, 5),
            "served_ntn_mbps": round(total_served_ntn * SCALE_FACTOR, 5)
        })
    df_summary = pd.DataFrame(summary_data)

st.success(f"Simulation Complete. Built {len(valid_tn_towers)} Terrestrial Towers.")

# ==========================================
# DISPLAY TABS WITH PLOTLY (Light/Academic Theme)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["Network Map", "Mobility and Density", "Traffic Analytics", "Data Export"])

bx, by = get_boundary_coords(ONTARIO_GEOM)
center_lat = (LAT_MIN + LAT_MAX) / 2
center_lon = (LON_MIN + LON_MAX) / 2

with tab1:
    st.subheader("Network Coverage: Terrestrial vs LEO Hexagonal Cells")
    fig1 = go.Figure()
    
    # Black border for light map
    fig1.add_trace(go.Scattermap(lat=by, lon=bx, mode='lines', line=dict(color='black', width=1.0), name='Ontario Boundary', hoverinfo='skip'))
    
    hx, hy = get_hex_grid_coords(hex_radius=0.6)
    fig1.add_trace(go.Scattermap(lat=hy, lon=hx, mode='lines', line=dict(color='magenta', width=1.5), opacity=0.4, name=f'LEO Hexagons (H3 Res {H3_RESOLUTION})', hoverinfo='skip'))

    tn_users_plot = np.array([[u.history_lat[0], u.history_lon[0]] for u in users if u.coverage_type == "TN"])
    leo_users_plot = np.array([[u.history_lat[0], u.history_lon[0]] for u in users if u.coverage_type == "LEO"])

    if len(leo_users_plot) > 0:
        fig1.add_trace(go.Scattermap(lat=leo_users_plot[:, 0], lon=leo_users_plot[:, 1], mode='markers', marker=dict(color='green', size=4, opacity=0.7), name='LEO Users'))
    if len(tn_users_plot) > 0:
        fig1.add_trace(go.Scattermap(lat=tn_users_plot[:, 0], lon=tn_users_plot[:, 1], mode='markers', marker=dict(color='blue', size=4, opacity=0.4), name='TN Users'))

    if len(valid_tn_towers) > 0:
        fig1.add_trace(go.Scattermap(lat=valid_tn_towers[:, 0], lon=valid_tn_towers[:, 1], mode='markers', marker=dict(color='red', size=10, opacity=0.9), name='Active TN Towers'))

    # Black text with a semi-transparent white background
    fig1.add_trace(go.Scattermap(lat=[c[0] for c in city_centers], lon=[c[1] for c in city_centers], mode='text', text=city_names, textposition='bottom right', textfont=dict(color='black', size=13), name='Cities', hoverinfo='skip'))

    fig1.update_layout(
        map=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=4.5),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        paper_bgcolor="white"
    )
    st.plotly_chart(fig1, width="stretch")

with tab2:
    st.subheader("STEPS Human Mobility")
    colA, colB = st.columns(2)
    
    with colA:
        fig2A = go.Figure()
        fig2A.add_trace(go.Scattermap(lat=by, lon=bx, mode='lines', line=dict(color='black', width=1.0), showlegend=False, hoverinfo='skip'))
        
        tracked_users = random.sample(users, 8)
        plot_colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'green']
        
        for idx, u in enumerate(tracked_users):
            fig2A.add_trace(go.Scattermap(lat=u.history_lat, lon=u.history_lon, mode='lines+markers', line=dict(color=plot_colors[idx]), marker=dict(size=5), name=f"User {u.user_id}"))
            
        fig2A.add_trace(go.Scattermap(lat=[c[0] for c in city_centers], lon=[c[1] for c in city_centers], mode='text', text=city_names, textposition='bottom right', textfont=dict(color='black', size=11), showlegend=False, hoverinfo='skip'))
        
        fig2A.update_layout(title="User Trajectories Over Time", map=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=4.2), height=550, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
        st.plotly_chart(fig2A, width="stretch")

    with colB:
        fig2B = go.Figure()
        fig2B.add_trace(go.Scattermap(lat=by, lon=bx, mode='lines', line=dict(color='black', width=1.0), showlegend=False, hoverinfo='skip'))
        
        all_lats, all_lons = [], []
        for u in users:
            all_lats.extend(u.history_lat)
            all_lons.extend(u.history_lon)
            
        # Changed the density points back to black
        fig2B.add_trace(go.Scattermap(lat=all_lats, lon=all_lons, mode='markers', marker=dict(color='black', size=3, opacity=0.03), showlegend=False))
        fig2B.add_trace(go.Scattermap(lat=[c[0] for c in city_centers], lon=[c[1] for c in city_centers], mode='text', text=city_names, textposition='bottom right', textfont=dict(color='black', size=11), showlegend=False, hoverinfo='skip'))
        
        fig2B.update_layout(title="Spatial Attractor Density", map=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=4.2), height=550, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
        st.plotly_chart(fig2B, width="stretch")

with tab3:
    st.subheader("24-Hour Network Traffic Profile")
    st.markdown("Visualizing the continuous Multi-Gaussian diurnal traffic wave.")
    
    # Plotly Line Chart for Traffic (Light Mode)
    fig3 = go.Figure()
    
    # Total Demand (Dark Blue)
    fig3.add_trace(go.Scatter(x=df_summary['Time Step'], y=df_summary['demand_mbps'], mode='lines', name='Total Demand (Mbps)', line=dict(color='#000080', width=3)))
    # TN Served (Standard Blue)
    fig3.add_trace(go.Scatter(x=df_summary['Time Step'], y=df_summary['served_tn_mbps'], mode='lines', name='Served by TN', line=dict(color='#1E90FF', width=2, dash='dash')))
    # LEO Overflow (Red/Magenta)
    fig3.add_trace(go.Scatter(x=df_summary['Time Step'], y=df_summary['served_ntn_mbps'], mode='lines', name='Overflow to LEO', line=dict(color='#D2042D', width=2, dash='dot')))
    
    fig3.update_layout(
        xaxis_title="Time of Day", 
        yaxis_title="Data Load (Mbps)", 
        template="plotly_white",  # Changed to clean white background
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig3.update_xaxes(tickformat="%H:%M", showgrid=True, gridcolor='lightgrey')
    fig3.update_yaxes(showgrid=True, gridcolor='lightgrey')
    
    st.plotly_chart(fig3, width="stretch")

with tab4:
    st.subheader("Data Exports")
    st.markdown("Raw data generated by the simulation engine.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. User Movement & Demand Data**")
        st.dataframe(df_users, width="stretch")
        csv1 = df_users.to_csv(index=False).encode('utf-8')
        st.download_button("Download simulated_users.csv", data=csv1, file_name="simulated_users_data_with_STEPS.csv", mime="text/csv")
    with col2:
        st.markdown("**2. 30-Min System Summary**")
        st.dataframe(df_summary, width="stretch")
        csv2 = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button("Download system_summary.csv", data=csv2, file_name="system_summary_table.csv", mime="text/csv")