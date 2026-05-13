import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from sklearn.cluster import KMeans

# ==========================================
# 1. DEFINE THE USER CLASS
# ==========================================
class User:
    def __init__(self, user_id, lat, lon):
        self.user_id = user_id
        self.lat = lat
        self.lon = lon
        self.base_demand_mbps = np.random.uniform(1.0, 5.0)
        
        self.coverage_type = None  # 'TN' or 'LEO'
        self.tn_cell_id = None     
        
    def get_demand_at_time(self, hour):
        time_multiplier = 1.0 + 0.5 * np.sin(np.pi * (hour - 14) / 12)
        return self.base_demand_mbps * time_multiplier

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

# Generate Dense City Users
for i in range(num_city_users):
    center = city_centers[np.random.randint(0, len(city_centers))]
    lat = np.random.normal(center[0], 0.15)
    lon = np.random.normal(center[1], 0.15)
    users.append(User(user_id=i, lat=lat, lon=lon))

# Generate Sparse Rural Users 
for i in range(num_city_users, num_city_users + num_rural_users):
    lat = np.random.uniform(42.5, 46.0)
    lon = np.random.uniform(-82.0, -75.0)
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

# ==========================================
# 4. EXPORT DATA TO CSV
# ==========================================
# (Data logic remains identical - still exports nicely for your team)
time_steps =[0, 1, 2, 3, 4, 5]
user_data_export =[]

for u in users:
    row_data = {
        'User_ID': u.user_id,
        'Latitude': round(u.lat, 4),
        'Longitude': round(u.lon, 4),
        'Coverage_Type': u.coverage_type,
        'Assigned_TN_Cell': u.tn_cell_id,
        'Base_Demand_Mbps': round(u.base_demand_mbps, 2)
    }
    for hour in time_steps:
        row_data[f'Demand_Mbps_Hour_{hour}'] = round(u.get_demand_at_time(hour), 2)
    user_data_export.append(row_data)

pd.DataFrame(user_data_export).to_csv("simulated_users_data_with_LEO.csv", index=False)

# ==========================================
# 5. VISUALIZE THE RESULTS (NOW WITH HEXAGONS)
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))

# --- NEW: DRAW NTN HEXAGONAL GRID ---
# Hexagon math setup
hex_radius = 0.4  # Size of the hex cell
horiz_spacing = np.sqrt(3) * hex_radius
vert_spacing = 1.5 * hex_radius

lon_min, lon_max = -82.5, -74.5
lat_min, lat_max = 42.0, 46.5

rows = int((lat_max - lat_min) / vert_spacing) + 2
cols = int((lon_max - lon_min) / horiz_spacing) + 2

# Generate the grid of hexagons
for row in range(rows):
    for col in range(cols):
        y = lat_min + row * vert_spacing
        x = lon_min + col * horiz_spacing
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
tn_users = np.array([[u.lat, u.lon] for u in users if u.coverage_type == 'TN'])
leo_users = np.array([[u.lat, u.lon] for u in users if u.coverage_type == 'LEO'])

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
ax.set_xlim([lon_min, lon_max])
ax.set_ylim([lat_min, lat_max])

ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()