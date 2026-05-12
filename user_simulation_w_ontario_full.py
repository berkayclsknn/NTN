import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from sklearn.cluster import KMeans
import random
import yaml

from shapely.geometry import shape, Point
from shapely.prepared import prep


# ==========================================
# 0. LOAD ONTARIO COORDINATE DATA FROM YAML
# ==========================================

ONTARIO_YAML_PATH = "E:\\berkay\\NTN\\ontario_full.yaml"

with open(ONTARIO_YAML_PATH, "r", encoding="utf-8") as f:
    ontario_yaml = yaml.safe_load(f)

ONTARIO_NAME = ontario_yaml.get("name", "Ontario")
H3_RESOLUTION = ontario_yaml.get("h3_resolution", None)

# Convert GeoJSON-style geometry from YAML into a Shapely geometry
ONTARIO_GEOM = shape(ontario_yaml["geojson_geometry"])
ONTARIO_PREPARED = prep(ONTARIO_GEOM)

# Shapely bounds order:
# minx = lon_min, miny = lat_min, maxx = lon_max, maxy = lat_max
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = ONTARIO_GEOM.bounds

print(f"Loaded geometry: {ONTARIO_NAME}")
print(f"H3 resolution from YAML: {H3_RESOLUTION}")
print(f"Ontario bounds:")
print(f"Longitude: {LON_MIN:.4f} to {LON_MAX:.4f}")
print(f"Latitude:  {LAT_MIN:.4f} to {LAT_MAX:.4f}")


def random_point_inside_ontario():
    """
    Generates one random point inside the Ontario polygon from the YAML file.
    Returns:
        lat, lon
    """
    while True:
        lon = np.random.uniform(LON_MIN, LON_MAX)
        lat = np.random.uniform(LAT_MIN, LAT_MAX)

        point = Point(lon, lat)

        if ONTARIO_PREPARED.contains(point):
            return lat, lon


def draw_ontario_boundary(ax, geom, edgecolor="black", linewidth=1.0, alpha=0.8, zorder=1):
    """
    Draws the Ontario boundary from the YAML MultiPolygon geometry.
    """
    polygons = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]

    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)


# ==========================================
# 1. REAL-WORLD ONTARIO TRAFFIC DATA
# ==========================================

BIG_CITY_TB = [98, 68, 53, 45, 45, 60, 91, 129, 151, 159, 163, 166,
               174, 169, 166, 174, 189, 204, 219, 242, 257, 249, 212, 159]

RURAL_TB = [19, 13, 10, 9, 9, 12, 18, 25, 29, 31, 32, 32,
            34, 33, 32, 34, 36, 39, 42, 47, 50, 48, 41, 31]

AVG_CITY_TB = sum(BIG_CITY_TB) / 24.0
AVG_RURAL_TB = sum(RURAL_TB) / 24.0


# ==========================================
# 2. DEFINE THE USER CLASS
# ==========================================

class User:
    def __init__(self, user_id, lat, lon):
        self.user_id = user_id

        self.home_lat = lat
        self.home_lon = lon

        self.attractors = [
            (self.home_lat, self.home_lon),
            (
                self.home_lat + np.random.normal(0, 0.1),
                self.home_lon + np.random.normal(0, 0.1)
            ),
            (
                self.home_lat + np.random.normal(0, 0.05),
                self.home_lon + np.random.normal(0, 0.05)
            )
        ]

        self.attractor_probs = [0.7, 0.2, 0.1]

        self.lat = self.home_lat
        self.lon = self.home_lon

        self.history_lat = [self.lat]
        self.history_lon = [self.lon]

        self.base_demand_mbps = np.random.uniform(1.0, 5.0)
        self.coverage_type = None
        self.tn_cell_id = None

    def get_demand_at_time(self, hour):
        """
        Calculates demand using the Ontario traffic profile.
        """

        is_city = True if self.coverage_type in ["TN", None] else False

        if is_city:
            hourly_multiplier = BIG_CITY_TB[hour] / AVG_CITY_TB
        else:
            hourly_multiplier = RURAL_TB[hour] / AVG_RURAL_TB

        return self.base_demand_mbps * hourly_multiplier

    def update_location_steps(self, hour):
        """
        STEPS Mobility Engine.
        """

        move_chance = 0.1 if (hour < 6 or hour > 22) else 0.4

        if np.random.rand() < move_chance:
            chosen_idx = np.random.choice(len(self.attractors), p=self.attractor_probs)
            target_lat, target_lon = self.attractors[chosen_idx]

            self.lat = target_lat + np.random.normal(0, 0.005)
            self.lon = target_lon + np.random.normal(0, 0.005)

        self.history_lat.append(self.lat)
        self.history_lon.append(self.lon)


# ==========================================
# 3. GENERATE USERS & LOCATIONS
# ==========================================

np.random.seed(42)

num_city_users = 700
num_rural_users = 300
users = []

city_centers = [
    (43.65, -79.38),  # Toronto
    (45.42, -75.69),  # Ottawa
    (43.25, -79.87),  # Hamilton
    (42.98, -81.25),  # London
    (44.23, -76.49),  # Kingston
    (46.49, -81.01)   # Sudbury
]

# Generate dense city users
for i in range(num_city_users):
    center = city_centers[np.random.randint(0, len(city_centers))]

    lat = np.random.normal(center[0], 0.15)
    lon = np.random.normal(center[1], 0.15)

    users.append(User(user_id=i, lat=lat, lon=lon))

# Generate sparse rural users inside real Ontario geometry from YAML
for i in range(num_city_users, num_city_users + num_rural_users):
    lat, lon = random_point_inside_ontario()
    users.append(User(user_id=i, lat=lat, lon=lon))

coordinates = np.array([[u.lat, u.lon] for u in users])


# ==========================================
# 4. CLUSTERING & LEO THRESHOLD LOGIC
# ==========================================

num_clusters = int(len(users) / 40)
TN_POP_THRESHOLD = 50
TN_BS_CAPACITY_MBPS = 10000

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coordinates)
potential_tn_locations = kmeans.cluster_centers_

cluster_counts = pd.Series(cluster_labels).value_counts()
valid_tn_towers = []

for i, user in enumerate(users):
    cluster_id = cluster_labels[i]
    cluster_population = cluster_counts[cluster_id]

    if cluster_population > TN_POP_THRESHOLD:
        user.coverage_type = "TN"
        user.tn_cell_id = cluster_id
    else:
        user.coverage_type = "LEO"
        user.tn_cell_id = "N/A (LEO)"

for cluster_id, count in cluster_counts.items():
    if count > TN_POP_THRESHOLD:
        valid_tn_towers.append(potential_tn_locations[cluster_id])

valid_tn_towers = np.array(valid_tn_towers)

print(f"Total TN Towers built (Passed threshold): {len(valid_tn_towers)}")


# ==========================================
# 5. EXPORT DATA TO CSV
# ==========================================

time_steps = list(range(24))
user_data_export = []

for u in users:
    row_data = {
        "User_ID": u.user_id,
        "Home_TN_Cell": u.tn_cell_id,
        "Base_Demand_Mbps": round(u.base_demand_mbps, 2)
    }

    for hour in time_steps:
        row_data[f"Demand_Hour_{hour}"] = round(u.get_demand_at_time(hour), 2)

        u.update_location_steps(hour)

        row_data[f"Lat_Hour_{hour}"] = round(u.lat, 4)
        row_data[f"Lon_Hour_{hour}"] = round(u.lon, 4)

    user_data_export.append(row_data)

pd.DataFrame(user_data_export).to_csv("simulated_users_data_with_STEPS.csv", index=False)
print("✅ Saved moving users to CSV!")


# ==========================================
# 6. VISUALIZE THE RESULTS
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ---------------------------------------------------------
# LEFT PLOT: User Trajectories Over Time
# ---------------------------------------------------------

tracked_users = random.sample(users, 8)
colors = plt.cm.tab10.colors

for idx, u in enumerate(tracked_users):
    ax1.plot(
        u.history_lon,
        u.history_lat,
        color=colors[idx % 10],
        marker="o",
        markersize=5,
        linestyle="--",
        linewidth=1.5,
        label=f"User {u.user_id}",
        zorder=3
    )

draw_ontario_boundary(ax1, ONTARIO_GEOM, edgecolor="black", linewidth=0.8, alpha=0.6, zorder=1)

ax1.set_title("User Trajectories Over Time", fontsize=14)
ax1.set_xlabel("Longitude", fontsize=12)
ax1.set_ylabel("Latitude", fontsize=12)

ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), shadow=True)
ax1.grid(True, linestyle="--", alpha=0.5)

# ---------------------------------------------------------
# RIGHT PLOT: Spatial Attractor Density
# ---------------------------------------------------------

all_lats = []
all_lons = []

for u in users:
    all_lats.extend(u.history_lat)
    all_lons.extend(u.history_lon)

ax2.scatter(
    all_lons,
    all_lats,
    c="black",
    s=10,
    alpha=0.02,
    marker="s",
    zorder=3
)

draw_ontario_boundary(ax2, ONTARIO_GEOM, edgecolor="black", linewidth=0.8, alpha=0.6, zorder=1)

ax2.set_title("Spatial Attractor Density (STEPS Output)", fontsize=14)
ax2.set_xticks([])
ax2.set_yticks([])

# Southern Ontario view, same as your original code
ax1.set_xlim([LON_MIN, LON_MAX])
ax1.set_ylim([LAT_MIN, LAT_MAX])
ax2.set_xlim([LON_MIN, LON_MAX])
ax2.set_ylim([LAT_MIN, LAT_MAX])

plt.tight_layout(pad=3.0)
plt.show()


# ==========================================
# 7. VISUALIZE THE RESULTS WITH HEXAGONS
# ==========================================

fig, ax = plt.subplots(figsize=(14, 12))

hex_radius = 0.6
horiz_spacing = np.sqrt(3) * hex_radius
vert_spacing = 1.5 * hex_radius

rows = int((LAT_MAX - LAT_MIN) / vert_spacing) + 2
cols = int((LON_MAX - LON_MIN) / horiz_spacing) + 2

# Generate hexagonal grid only over Ontario
for row in range(rows):
    for col in range(cols):
        y = LAT_MIN + row * vert_spacing
        x = LON_MIN + col * horiz_spacing

        if row % 2 == 1:
            x += horiz_spacing / 2

        # Only draw hexagons whose center is inside Ontario
        if not ONTARIO_PREPARED.contains(Point(x, y)):
            continue

        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_radius,
            orientation=0,
            facecolor="none",
            edgecolor="magenta",
            alpha=0.4,
            linewidth=1.5,
            zorder=2
        )

        ax.add_patch(hex_patch)

ax.plot([], [], color="magenta", label=f"Fixed NTN Hexagonal Cells, YAML H3 Res {H3_RESOLUTION}")

# --- PLOT USERS & TOWERS ---

tn_users = np.array([
    [u.history_lat[0], u.history_lon[0]]
    for u in users
    if u.coverage_type == "TN"
])

leo_users = np.array([
    [u.history_lat[0], u.history_lon[0]]
    for u in users
    if u.coverage_type == "LEO"
])

if len(leo_users) > 0:
    ax.scatter(
        leo_users[:, 1],
        leo_users[:, 0],
        c="green",
        s=15,
        alpha=0.6,
        label="LEO Users (Low Density)",
        zorder=4
    )

if len(tn_users) > 0:
    ax.scatter(
        tn_users[:, 1],
        tn_users[:, 0],
        c="blue",
        s=15,
        alpha=0.3,
        label="TN Users (High Density)",
        zorder=4
    )

if len(valid_tn_towers) > 0:
    ax.scatter(
        valid_tn_towers[:, 1],
        valid_tn_towers[:, 0],
        c="red",
        marker="^",
        s=150,
        edgecolor="black",
        label="Active TN Towers",
        zorder=5
    )

for center, name in zip(
    city_centers,
    ["Toronto", "Ottawa", "Hamilton", "London", "Kingston", "Sudbury"]
):
    ax.text(
        center[1],
        center[0],
        name,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        zorder=6
    )

# Draw Ontario boundary from YAML
draw_ontario_boundary(ax, ONTARIO_GEOM, edgecolor="black", linewidth=1.0, alpha=0.9, zorder=3)

ax.set_title("Network Coverage: Terrestrial Network vs LEO Hexagonal Cells", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_xlim([LON_MIN, LON_MAX])
ax.set_ylim([LAT_MIN, LAT_MAX])

ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================
# 8. GENERATE SYSTEM SUMMARY TABLE
# ==========================================

print("Generating System Summary Table (Slide 2 format)...")

LEO_TOTAL_CAPACITY_MBPS = 15000
SCALE_FACTOR = 500

time_series = pd.date_range(
    start="2025-01-01 00:00",
    end="2025-01-01 23:30",
    freq="30min"
)

summary_data = []

for ts in time_series:
    current_hour = ts.hour

    total_demand = 0.0
    total_served_tn = 0.0
    total_served_ntn = 0.0

    tn_tower_loads = {tower_id: 0.0 for tower_id in range(num_clusters)}
    leo_total_load = 0.0

    for u in users:
        user_demand = u.get_demand_at_time(current_hour)
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

df_summary.to_csv("system_summary_table.csv", index=False)

print("✅ Saved System Summary to 'system_summary_table.csv'")
print("\n--- Preview of System Summary ---")
print(df_summary.head(10).to_string(index=False))