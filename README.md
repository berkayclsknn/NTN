🛰️ Hybrid TN & LEO Network Digital Twin

📖 Overview

This project is a Python-based Digital Twin Simulation Engine for a hybrid
telecommunications network comprising Terrestrial Networks (TN) and Low Earth
Orbit (LEO) Satellite Networks (NTN).

The simulation models human spatio-temporal mobility, heterogeneous 5G data
demand, and geographic cell-tower clustering across the province of Ontario,
Canada. It serves as the foundational data generation layer for downstream
Machine Learning (CAPEX/OPEX Optimization) and Satellite Constellation
(Walker-Delta SINR) modeling.

✨ Key Features & Academic Foundation

1. Spatio-Temporal Human Mobility (The STEPS Model)

Instead of relying on random waypoint models, simulated users are governed by
the STEPS (Spatio-TEmporal Parametric Stepping) methodology. Individual human
movement is mathematically modeled using:

  - Visitation Probability (Zipf's Law): Users have primary, secondary, and
    tertiary "Attractors" (Home, Work, Social). Visitation probability decays
    following a power-law distribution.
  - Displacement Distance (Truncated Power-Law): The distance of a user's
    commute is calculated via mathematical Rejection Sampling to mirror
    real-world heavy-tailed human travel patterns.

📚 Reference: González, M. C., Hidalgo, C. A., & Barabási, A. L. (2008).
"Understanding individual human mobility patterns." Nature.

2. Heterogeneous Traffic Profiles & Diurnal Modeling

To accurately stress-test the network, user data demand is generated dynamically
rather than homogeneously:

  - 3GPP User Profiles: The population is divided into a 20/60/20 stochastic
    distribution. 20% are Heavy Users (eMBB / 4K Video), 60% are Medium Users
    (Nominal Broadband), and 20% are Light Users (mMTC / Texting).
  - Diurnal Traffic Curve: Daily traffic variations (e.g., the lunchtime bump
    and the 8:00 PM streaming peak) are modeled using a continuous Sum of
    Gaussians (Multi-Gaussian) mathematical function to simulate the human
    sleep/wake cycle.

📚 Reference: 3GPP TR 38.913 (Study on Scenarios and Requirements for Next
Generation Access Technologies).

3. Data-Driven Network Architecture (K-Means)

The simulation dynamically plans the telecommunications infrastructure using
geographic density:

  - TN Deployment: Uses K-Means clustering to find dense population centers
    (e.g., Toronto, Ottawa). If a cluster exceeds the TN_POP_THRESHOLD of 50
    users, a 10 Gbps Terrestrial Base Station is built to serve it.
  - LEO Fallback: Rural users who fail the density threshold (or users whose TN
    towers exceed the 10 Gbps capacity limit) are automatically routed to the
    Non-Terrestrial Network (Starlink/LEO Hexagons).

⚙️ Dependencies & Installation

This project requires Python 3.8+ and the following libraries. You can install
them via pip:

pip install numpy pandas matplotlib scikit-learn shapely pyyaml

Required Files:

  - user_simulation_w_ontario_full.py (Main Script)
  - ontario_full.yaml (Contains the GeoJSON polygon boundaries and H3 Resolution
    configurations for the map).

🚀 How to Run

Execute the main script from your terminal:

python user_simulation_w_ontario_full.py

Upon execution, the script will:

1.  Parse the Ontario boundaries.
2.  Generate 1,000 scale-model users.
3.  Calculate TN cluster placements and route traffic.
4.  Export the resulting datasets to CSV.
5.  Render two visual Matplotlib windows.

📊 Outputs Generated

1. Data Deliverables (CSVs)

  - simulated_users_data_with_STEPS.csv
      - Purpose: For the Satellite Team.
      - Description: Tracks the exact Latitude, Longitude, and Mbps demand of
        every individual user for all 24 hours. Used to calculate moving
        satellite beam coverage and SINR.
  - system_summary_table.csv
      - Purpose: For the Machine Learning / Optimization Team.
      - Description: Aggregates the total network load in 30-minute intervals.
        Displays total demand, data successfully served by TN, and overflow data
        pushed to LEO. The data is multiplied by a SCALE_FACTOR to represent
        millions of Mbps for province-wide optimization.

2. Visual Deliverables (Plots)

  - Figure 1: Behavioral & Mobility Proof 
      - Left: Trajectories of tracked users commuting across the province.
      - Right: Spatial Attractor Density cloud, visually proving the power-law
        clustering of human populations.
  - Figure 2: Network Coverage Map 
      - A scaled geographic map of Ontario showing dense TN users (Blue), rural
        LEO users (Green), active TN Towers (Red Triangles), and the fixed H3
        Resolution LEO Hexagonal grid (Pink).

🗺️ Next Steps / Phase 2

  - Graphical User Interface (GUI): A front-end dashboard (e.g., Streamlit or
    Tkinter) to allow non-technical users to dynamically adjust variables like
    Tower Capacity, Population Counts, and Traffic Multipliers without editing
    the raw Python code.
  - Dynamic Handover: Implementing active distance-checks to allow users to
    dynamically switch between TN and LEO coverage as they physically commute
    out of range of their home cell tower.
