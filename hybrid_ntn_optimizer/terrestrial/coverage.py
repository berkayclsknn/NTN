import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import List
from omegaconf import DictConfig
import math

# Backend physics and utility imports
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation # Ensure name matches your init

def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    print("[AREA-MODE] Strategy: Pure Cluster-Membership (Voronoi) Coverage")
    
    # --- STAGE 1: URBAN DISCOVERY (Threshold = 50) ---
    all_coords = np.array([[u.home_lat, u.home_lon] for u in users])
    k_discovery = max(1, int(len(users) / 50)) 
    kmeans_discovery = KMeans(n_clusters=k_discovery, random_state=cfg.random_seed, n_init=10)
    discovery_labels = kmeans_discovery.fit_predict(all_coords)
    
    user_df = pd.DataFrame({
        'discovery_label': discovery_labels,
        'user_obj': users
    })
    
    cluster_counts = user_df['discovery_label'].value_counts()
    urban_cluster_ids = cluster_counts[cluster_counts >= 50].index.tolist()
    
    base_stations = []
    bs_id_counter = 0

    # --- STAGE 2: VORONOI AREA PARTITIONING ---
    for zone_id in urban_cluster_ids:
        zone_users = user_df[user_df['discovery_label'] == zone_id]
        zone_coords = np.array([[u.user_obj.home_lat, u.user_obj.home_lon] for _, u in zone_users.iterrows()])
        
        # Target: 20 users per tower Area
        k_for_zone = math.ceil(len(zone_users) / 20)
        
        kmeans_final = KMeans(n_clusters=k_for_zone, random_state=cfg.random_seed, n_init=10)
        final_labels = kmeans_final.fit_predict(zone_coords)
        centers = kmeans_final.cluster_centers_
        
        for i in range(k_for_zone):
            c_lat, c_lon = centers[i]
            
            # We set radius to effectively 0 to remove GUI circles.
            # Membership is now the ONLY way to connect.
            bs = BaseStation(
                bs_id=bs_id_counter, 
                lat=c_lat, lon=c_lon, 
                capacity_mbps=cfg.terrestrial.bs_capacity_mbps,
                total_bandwidth_hz=cfg.terrestrial.bandwidth_hz,
                use_physical_radius=False, # DISBALE RADIUS CHECK
                coverage_radius_km=0.01    # HIDDEN IN GUI
            )
            bs.set_resolution(h3_resolution)
            base_stations.append(bs)
            
            # HARD-LINK: Every user in this K-Means "Area" is OWNED by this BS
            assigned_indices = np.where(final_labels == i)[0]
            for idx_in_zone in assigned_indices:
                user_obj = zone_users.iloc[idx_in_zone]['user_obj']
                user_obj.tn_cell_id = bs_id_counter
                user_obj.coverage_type = "TN" # Force coverage because they are in the area
            
            bs_id_counter += 1

    # --- RURAL HANDOFF ---
    rural_users = user_df[~user_df['discovery_label'].isin(urban_cluster_ids)]
    for _, r_row in rural_users.iterrows():
        r_row['user_obj'].coverage_type = "DROPPED" # Logic will re-route to LEO
        r_row['user_obj'].tn_cell_id = -1

    print(f"Final Deployment: {len(base_stations)} Area-Based Towers.")
    return base_stations