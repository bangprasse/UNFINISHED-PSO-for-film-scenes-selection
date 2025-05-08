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
print_df("Data Source for Total Cost Between All Scenes", Cost_df)

# Change column name same as index
Cost_df.columns = range(Cost_df.shape[1])
Cost_df = Cost_df.reset_index()
Cost_df.drop(["index"], inplace=True, axis=1)
print_df("Converted Cost_df", Cost_df)


# PSO INITIALIZATION
# -------------------------------------------
# Printing The Section
print("############################################")
print("         SECTION: PSO INITIALIZATION")
print("############################################")
print("")

# Base Parameter
n = 100  # Max Iteration
N = 25  # Swarm Size
d = len(scene_name)  # Dimention Size
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
V_max = round((rd.uniform(0, 1)) * (X_max - X_min), 6)
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

# Execute PSO algorithm
Result = PSO_exe(
    Result, Cost_df, n, N, d, particle_list, c1, c2, X_min, X_max, V_min, V_max
)

# Printing All Dataframe Output
print_df("Output: Dataframe for Position", Result[0])
print_df("Output: Dataframe for Inertia Weight", Result[7])
print_df("Output: Dataframe for Velocity", Result[1])
print_df("Output: Dataframe for Route", Result[2])
print_df("Output: Dataframe for Cost", Result[3])
print_df("Output: Dataframe for Fitness Value", Result[4])
print_df("Output: Dataframe for Pbest", Result[5])
print_df("Output: Dataframe for Gbest", Result[6])

# Printing Summary
print("==================== PSO Result Summary ====================")
print(">> Swarm Size (N)    : ", N)
print(">> Max Iteration (n) : ", n)
print(">> Cognitive (c1)    : ", c1)
print(">> Social (c2)       : ", c2)
print(">> Inertia Weight    : Random Inertia Weight Strategy")
print("")
print(">> Best Route        : ")
print(">> Best Cost         : ")
