# %reset -f

# IMPORT PACKAGES, /modules, AND LIBRARIES
# -------------------------------------------
# Import functions from modules
from modules import *

# Import all data sources from data_input.py
from datasource_input import *

# clearscreen()
# DATA SOURCE INIT AND PREPROCCESING
# -------------------------------------------
# Data sources and cases parameter initialization

# Printing The Section
print("############################################")
print("         SECTION: DATA SOURCES USED")
print("############################################")
print("")

# Car usage
print("Number of Cars: ", Cars)

# 1. Data Source for All Scenes
#    dataframe name = AllScn_df
print_df("Data for All Scenes", Scn_df)

# 2. Data Source for The Distance Between All Locations
#    dataframe name = LocDis_df
print_df("Data for The Distance Between All Locations", Loc_df)

# 3. Data Source for Cost per Scene of Each Talent
print_df("Data for Cost per Scene of Each Talent", Tal_df)

# 4. Data Source for Total Talent Cost per Scene
print_df("Data Source for Total Talent Cost per Scene", Tal_Cos_df)

# # 5. Data Source for Distance Between All Scenes
print_df("Data Source for Distance Between All Scenes", Scn_Dis_df)

# # 6. Data Source for Fuel Cost Between All Scenes
print_df("Data Source for Fuel Cost Between All Scenes", Fuel_Cost_df)

# # 7. Data Source for Total Cost Between All Scenes
print_df("Data Source for Total Cost Between All Scenes", norm_Cost_df)

# Get Max and Min Value of Cost_df before Normalization
max_value = max_val
min_value = min_val

# Change column name same as index
norm_Cost_df.columns = range(norm_Cost_df.shape[1])
norm_Cost_df = norm_Cost_df.reset_index()
norm_Cost_df.drop(["index"], inplace=True, axis=1)
print_df("Converted Cost_df", norm_Cost_df)


# PSO INITIALIZATION
# -------------------------------------------
# Printing The Section
print("############################################")
print("         SECTION: PSO INITIALIZATION")
print("############################################")
print("")

# Base Parameter
n = 1500  # Max Iteration
N = 75  # Swarm Size
particle_list = ["Prt" + str(i + 1) for i in range(0, N)]  # List of Particle Names

c1 = 2  # Learning Rates
c2 = 2  # Learning Rates

# Inertia Weight
# here, i use random inertia weight strategy every iteration
# so, i generate it by a function in modules/function_def.py

# Position Clamping
X_min = 0
X_max = 1

# Velocity Clamping
V_max = round((rd.uniform(0, 1)) * (X_max - X_min), 7)
# V_max = round(0.5 * (X_max - X_min), 7)
V_min = -V_max

# Generate new dataframe
Xj = pd.DataFrame(columns=particle_list)  # Storage of position data in each iteration

V_col = ["r1", "r2"]
for particle in particle_list:
    V_col.append(particle)
Vj = pd.DataFrame(columns=V_col)  # Storage of velocity data in each iteration
Route_j = pd.DataFrame(columns=particle_list)  # Storage of the routes result
Cost_j = pd.DataFrame(columns=particle_list)  # Storage of the costs result
Fitness_j = pd.DataFrame(columns=particle_list)  # Storage of the fitness value
Pbest_j = pd.DataFrame(columns=particle_list)  # Storage of the Pbest in each iteration
Gbest_i = pd.DataFrame(
    columns=["Gbest", "Fitness Value"]
)  # Storage of the Gbest in each iteration
Inertia_df = pd.DataFrame(
    {"Inertia Weight": [0]}
)  # Storage of the Inertia Weight in each iteration

# Grouping Dataframe
Result = [Xj, Vj, Route_j, Cost_j, Fitness_j, Pbest_j, Gbest_i, Inertia_df]

# Defining the starting scene
starting_scene = 4

# Since the starting scene is fixed, the dimension equal the total number of scenes minus 1
d = len(scene_name) - 1  # Dimention Size

# Execute PSO algorithm
Result = PSO_exe(
    Result,
    norm_Cost_df,
    n,
    N,
    d,
    particle_list,
    c1,
    c2,
    X_min,
    X_max,
    V_min,
    V_max,
    starting_scene,
    scene_name,
)

# Printing Output
print_output(
    Result,
    norm_Cost_df,
    Cost_df,
    N,
    n,
    c1,
    c2,
    V_max,
    V_min,
    scene_name,
    max_value,
    min_value,
    starting_scene,
)

# Optimality Comparison of Routes
Starting_Route = [
    4,
    3,
    "6A",
    "6B",
    37,
    "35A",
    "35B",
    34,
    "21A",
    "21B",
    "14A",
    "14B",
    5,
    "1C",
    "22A",
    36,
    8,
    "9A",
    "9B",
    18,
    "30A",
    "30B",
    "30C",
    15,
    17,
    10,
    20,
    25,
    27,
    "16A",
    "16B",
    "16C",
    "16D",
    "29B",
    24,
    26,
    28,
    "7A",
    "7B",
    "12A",
    "12B",
    "32A",
    "32B",
    "29A",
    "22B",
    19,
    "19B",
    "1D",
    "22C",
    "23A",
    "23B",
    31,
    "33A",
    "33B",
    "34B",
    "2A",
    "2B",
    13,
    "1E",
    11,
]
Optimality_comparison(Starting_Route, Cost_df, n)


# G_df = Result[6]
# last_route = G_df["Gbest"][n]
# route = sorted(range(len(last_route)), key=lambda x: last_route[x])
# The_route = [scene_name[rout] for rout in route]

# cost_a = 0
# cost_b = 0
# for idx in range(0, len(The_route) - 1):

#     va1 = The_route[idx]
#     va2 = The_route[idx + 1]
#     vb1 = Starting_Route[idx]
#     vb2 = Starting_Route[idx + 1]

#     cost_a = round(cost_a + Cost_df[va2][va1], 6)
#     cost_b = round(cost_b + Cost_df[vb2][vb1], 6)

#     print(str(va2) + " # " + str(vb2) + " === " + str(cost_a) + " # " + str(cost_b))
