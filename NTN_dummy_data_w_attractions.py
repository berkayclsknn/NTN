import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from sklearn.cluster import KMeans
import random

# ==========================================
# 0. REAL-WORLD ONTARIO TRAFFIC DATA (TB/hour) AI created but still better than dummy data (closer to the original data)
# ==========================================
# Data from 00:00 to 23:00
BIG_CITY_TB =[98, 68, 53, 45, 45, 60, 91, 129, 151, 159, 163, 166, 
               174, 169, 166, 174, 189, 204, 219, 242, 257, 249, 212, 159]

RURAL_TB =[19, 13, 10, 9, 9, 12, 18, 25, 29, 31, 32, 32, 
            34, 33, 32, 34, 36, 39, 42, 47, 50, 48, 41, 31]

# Calculate 24-hour averages to create multipliers
AVG_CITY_TB = sum(BIG_CITY_TB) / 24.0  # ~151.7 TB/hr
AVG_RURAL_TB = sum(RURAL_TB) / 24.0    # ~29.0 TB/hr


# ==========================================
# 1. DEFINE THE USER CLASS
# ==========================================
class User:
    def __init__(self, user_id, lat, lon):
        self.user_id = user_id
        
        # --- STEPS MOBILITY: ATTRACTORS ---
        self.home_lat = lat
        self.home_lon = lon
        
        # We give each user 3 favorite places: Home, Work, and Social
        # Work is about ~10km away, Social is about ~5km away
        self.attractors =[
            (self.home_lat, self.home_lon),  # Index 0: Home
            (self.home_lat + np.random.normal(0, 0.1), self.home_lon + np.random.normal(0, 0.1)), # Index 1: Work
            (self.home_lat + np.random.normal(0, 0.05), self.home_lon + np.random.normal(0, 0.05)) # Index 2: Social
        ]
        
        # --- STEPS MOBILITY: PREFERENTIAL ATTACHMENT ---
        # 70% chance to go Home, 20% to go to Work, 10% to go Social
        self.attractor_probs =[0.7, 0.2, 0.1]
        
        # Current actual location starts at Home
        self.lat = self.home_lat
        self.lon = self.home_lon
        
        # --- NEW: MEMORY FOR PLOTTING ---
        # We start by remembering their starting location
        self.history_lat = [self.lat]
        self.history_lon = [self.lon]        
        
        self.base_demand_mbps = np.random.uniform(1.0, 5.0)
        self.coverage_type = None  
        self.tn_cell_id = None     
        
    def get_demand_at_time(self, hour):
        """
        Calculates demand using the exact Ontario real-world data profile!
        """
        # We need to know if they are a city user or rural user 
        # (Default to City if coverage_type hasn't been assigned yet)
        is_city = True if self.coverage_type in['TN', None] else False
        
        if is_city:
            hourly_multiplier = BIG_CITY_TB[hour] / AVG_CITY_TB
        else:
            hourly_multiplier = RURAL_TB[hour] / AVG_RURAL_TB
            
        # Multiply their personal baseline by the real-world hourly wave
        return self.base_demand_mbps * hourly_multiplier

    def update_location_steps(self, hour):
        """
        The STEPS Mobility Engine. Runs once every hour.
        """
        # People move less at night (0-5 AM) and more during the day
        move_chance = 0.1 if (hour < 6 or hour > 22) else 0.4
        
        # Roll a virtual dice to see if they travel this hour
        if np.random.rand() < move_chance:
            # Pick which attractor they go to based on the 70/20/10 probabilities
            chosen_idx = np.random.choice(len(self.attractors), p=self.attractor_probs)
            target_lat, target_lon = self.attractors[chosen_idx]
            
            # Move the user to the target, plus a tiny bit of random wandering
            self.lat = target_lat + np.random.normal(0, 0.005)
            self.lon = target_lon + np.random.normal(0, 0.005)

        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)            

# ==========================================
# 2. GENERATE USERS & LOCATIONS
# ==========================================
np.random.seed(42)
num_city_users = 700
num_rural_users = 300 
users =[]

city_centers =[
    (43.65, -79.38),  # Toronto
    (45.42, -75.69),  # Ottawa
    (43.25, -79.87),  # Hamilton
    (42.98, -81.25),  # London
    (44.23, -76.49),   # Kingston
    (46.49, -81.01)    #Sudbury
]

# --- NEW ONTARIO BOUNDARIES FROM TEAM ---
LON_MIN, LON_MAX = -95.0, -74.0
LAT_MIN, LAT_MAX = 41.7, 57.0

# Generate Dense City Users
for i in range(num_city_users):
    center = city_centers[np.random.randint(0, len(city_centers))]
    lat = np.random.normal(center[0], 0.15)
    lon = np.random.normal(center[1], 0.15)
    users.append(User(user_id=i, lat=lat, lon=lon))

# Generate Sparse Rural Users 
for i in range(num_city_users, num_city_users + num_rural_users):
    lat = np.random.uniform(LAT_MIN, LAT_MAX)
    lon = np.random.uniform(LON_MIN, LON_MAX)
    users.append(User(user_id=i, lat=lat, lon=lon))


coordinates = np.array([[u.lat, u.lon] for u in users])

# ==========================================
# 3. CLUSTERING & LEO THRESHOLD LOGIC
# ==========================================
# Force K-Means to average exactly 40 users per cluster
num_clusters = int(len(users) / 40)  
TN_POP_THRESHOLD = 50  
TN_BS_CAPACITY_MBPS = 10000

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coordinates)
potential_tn_locations = kmeans.cluster_centers_

cluster_counts = pd.Series(cluster_labels).value_counts()
valid_tn_towers =[] 

for i, user in enumerate(users):
    cluster_id = cluster_labels[i]
    cluster_population = cluster_counts[cluster_id]
    
    if cluster_population > TN_POP_THRESHOLD:
        user.coverage_type = 'TN'
        user.tn_cell_id = cluster_id
    else:
        user.coverage_type = 'LEO'
        user.tn_cell_id = 'N/A (LEO)'

for cluster_id, count in cluster_counts.items():
    if count > TN_POP_THRESHOLD:
        valid_tn_towers.append(potential_tn_locations[cluster_id])
        
valid_tn_towers = np.array(valid_tn_towers)
print(f"Total TN Towers built (Passed threshold): {len(valid_tn_towers)}")

# ==========================================
# 4. EXPORT DATA TO CSV
# ==========================================

time_steps = list(range(24)) 
user_data_export =[]

for u in users:
    # Basic info that doesn't change
    row_data = {
        'User_ID': u.user_id,
        'Home_TN_Cell': u.tn_cell_id, # Where they started their day
        'Base_Demand_Mbps': round(u.base_demand_mbps, 2)
    }
    
    # Run through the hours
    for hour in time_steps:
        # 1. Calculate their data demand for the hour
        row_data[f'Demand_Hour_{hour}'] = round(u.get_demand_at_time(hour), 2)
        
        # 2. Trigger the STEPS movement engine!
        u.update_location_steps(hour)
        
        # 3. Record their new GPS coordinates for this hour
        row_data[f'Lat_Hour_{hour}'] = round(u.lat, 4)
        row_data[f'Lon_Hour_{hour}'] = round(u.lon, 4)
        
    user_data_export.append(row_data)

pd.DataFrame(user_data_export).to_csv("simulated_users_data_with_STEPS.csv", index=False)
print("✅ Saved moving users to CSV!")

# ==========================================
# 5. VISUALIZE THE RESULTS (EXACT SLIDE 4 FORMAT)
# ==========================================
# Create a figure with 2 subplots side-by-side (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ---------------------------------------------------------
# LEFT PLOT: "User Trajectories Over Time"
# ---------------------------------------------------------
# Pick 8 random users to track, just like the slide's legend
tracked_users = random.sample(users, 8)
colors = plt.cm.tab10.colors  # Get distinct colors

for idx, u in enumerate(tracked_users):
    # Plot their path with dots and lines
    ax1.plot(u.history_lon, u.history_lat, color=colors[idx % 10], 
             marker='o', markersize=5, linestyle='--', linewidth=1.5, 
             label=f'User {u.user_id}')

ax1.set_title('User Trajectories Over Time', fontsize=14)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)

# Place the legend on the right side of the first plot, exactly like the slide
ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), shadow=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------
# RIGHT PLOT: Attractor / Density Cloud
# ---------------------------------------------------------
# To recreate the grayscale cluster plot, we need EVERY step from EVERY user
all_lats = []
all_lons =[]
for u in users:
    all_lats.extend(u.history_lat)
    all_lons.extend(u.history_lon)

# Scatter them as tiny, highly transparent black squares to get that exact pixelated look
ax2.scatter(all_lons, all_lats, c='black', s=10, alpha=0.02, marker='s')

ax2.set_title('Spatial Attractor Density (STEPS Output)', fontsize=14)

# Remove the axis ticks and labels to match the raw visual style of the slide's right image
ax2.set_xticks([])
ax2.set_yticks([])

# Make sure both maps are scaled identically
ax1.set_xlim([-82.5, -74.5])
ax1.set_ylim([42.0, 46.5])
ax2.set_xlim([-82.5, -74.5])
ax2.set_ylim([42.0, 46.5])

# Adjust spacing so the legend doesn't overlap the second plot
plt.tight_layout(pad=3.0)
plt.show()

# ==========================================
# VISUALIZE THE RESULTS (WITH HEXAGONS)
# ==========================================
# --- PLOT B: SLIDE 1 FORMAT (Hexagons & Towers) ---
fig, ax = plt.subplots(figsize=(14, 12)) # Taller figure because Ontario is huge!

# The team requested H3 Resolution 2 (~118km wide). 
# 1 degree lat = ~111km. A hex_radius of 0.6 perfectly mimics 118km width!
hex_radius = 0.6 
horiz_spacing = np.sqrt(3) * hex_radius
vert_spacing = 1.5 * hex_radius

# We use the new boundaries we defined in Section 2
for row in range(int((LAT_MAX - LAT_MIN) / vert_spacing) + 2):
    for col in range(int((LON_MAX - LON_MIN) / horiz_spacing) + 2):
        y = LAT_MIN + row * vert_spacing
        x = LON_MIN + col * horiz_spacing
        if row % 2 == 1: x += horiz_spacing / 2
        ax.add_patch(RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=0, facecolor='none', edgecolor='magenta', alpha=0.4, linewidth=1.5))

rows = int((LAT_MAX - LAT_MIN) / vert_spacing) + 2
cols = int((LON_MAX - LON_MIN) / horiz_spacing) + 2

# Generate the grid of hexagons
for row in range(rows):
    for col in range(cols):
        y = LAT_MIN + row * vert_spacing
        x = LON_MIN + col * horiz_spacing
        # Stagger every other row to interlock the hexagons
        if row % 2 == 1:
            x += horiz_spacing / 2
            
        # Draw hexagon (Magenta to match Slide 1)
        hex_patch = RegularPolygon((x, y), numVertices=6, radius=hex_radius, 
                                   orientation=0, facecolor='none', 
                                   edgecolor='magenta', alpha=0.4, linewidth=1.5)
        ax.add_patch(hex_patch)

# Trick to add the hexagon to the legend
ax.plot([],[], color='magenta', label='Fixed NTN Hexagonal Cells (LEO)')

# --- PLOT USERS & TOWERS ---
tn_users = np.array([[u.history_lat[0], u.history_lon[0]] for u in users if u.coverage_type == 'TN'])
leo_users = np.array([[u.history_lat[0],  u.history_lon[0]] for u in users if u.coverage_type == 'LEO'])

if len(leo_users) > 0:
    ax.scatter(leo_users[:, 1], leo_users[:, 0], c='green', s=15, alpha=0.6, label='LEO Users (Low Density)')

if len(tn_users) > 0:
    ax.scatter(tn_users[:, 1], tn_users[:, 0], c='blue', s=15, alpha=0.3, label='TN Users (High Density)')

if len(valid_tn_towers) > 0:
    ax.scatter(valid_tn_towers[:, 1], valid_tn_towers[:, 0], c='red', marker='^', s=150, edgecolor='black', label='Active TN Towers')

for center, name in zip(city_centers,['Toronto', 'Ottawa', 'Hamilton', 'London', 'Kingston', 'Sudbury']):
    ax.text(center[1], center[0], name, fontsize=12, fontweight='bold', ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# Finalize Plot Aesthetics
ax.set_title(f'Network Coverage: Terrestrial Network vs LEO Hexagonal Cells', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Set map boundaries so the hexagons don't stretch into nowhere
ax.set_xlim([LON_MIN, LON_MAX])
ax.set_ylim([LAT_MIN, LAT_MAX])

ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 5. GENERATE SYSTEM SUMMARY TABLE (SLIDE 2)
# ==========================================
print("Generating System Summary Table (Slide 2 format)...")

LEO_TOTAL_CAPACITY_MBPS = 15000 
SCALE_FACTOR = 500  # Scale up our 1,000 users to represent millions of Mbps

# Create time steps every 30 mins for Jan 1, 2025
time_series = pd.date_range(start="2025-01-01 00:00", end="2025-01-01 23:30", freq="30min")
summary_data =[]

for ts in time_series:
    current_hour = ts.hour
    total_demand = total_served_tn = total_served_ntn = 0.0
    tn_tower_loads = {tower_id: 0.0 for tower_id in range(num_clusters)}
    leo_total_load = 0.0
    
    # 1. Gather demand from all users for this specific time
    for u in users:
        user_demand = u.get_demand_at_time(current_hour)
        total_demand += user_demand
        if u.coverage_type == 'TN' and u.tn_cell_id != 'N/A':
            tn_tower_loads[u.tn_cell_id] += user_demand
        else:
            leo_total_load += user_demand
            
    # 2. Process TN Towers & Overflow (Slide 3 Rule)
    for tower_id, load in tn_tower_loads.items():
        if load <= TN_BS_CAPACITY_MBPS:
            total_served_tn += load
        else:
            total_served_tn += TN_BS_CAPACITY_MBPS
            leo_total_load += (load - TN_BS_CAPACITY_MBPS) # Overflow goes to LEO!
            
    # 3. Process LEO Capacity (Multiply by 0.98 to 1.02 to simulate orbital movement)
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
df_summary.to_csv("system_summary_table.csv", index=False)
print("✅ Saved System Summary to 'system_summary_table.csv'")
print("\n--- Preview of System Summary ---")
print(df_summary.head(10).to_string(index=False))

