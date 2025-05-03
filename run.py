# %reset -f

# IMPORT PACKAGES, /modules, AND LIBRARIES
# -------------------------------------------
# Import functions from modules
from modules import *

# clearscreen()
# DATA SOURCE INIT AND PREPROCCESING
# -------------------------------------------
# Input datasource to Master_df
Master_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data.xlsx",
        sheet_name="Location Distance",
    )
)
Master_df = Master_df.drop(Master_df.columns[0], axis=1)
print_df(Master_df)

# Get Master_df header column before change it
Headers = Master_df.columns.to_list()

# Change column name same as index
Master_df.columns = range(Master_df.shape[1])
print_df(Master_df)


# PSO INITIALIZATION
# -------------------------------------------
# Base Parameter
n = 100  # Max Iteration
N = 25  # Swarm Size
d = len(Master_df.columns)  # Dimention Size
particle_list = ["X" + str(i + 1) for i in range(0, N)]  # List of Particle Names

c1 = 2  # Learning Rates
c2 = 2  # Learning Rates

# Inertia Weight
# here, i use random inertia weight strategy every iteration
# so, i generate it by a function in modules/function_def.py

# Position Clamping
X_min = 0
X_max = 1

# Velocity Clamping
V_max = (rd.uniform(0, 1)) * (X_max - X_min)
V_min = -V_max

# Generate new dataframe
Xj = pd.DataFrame(columns=particle_list)  # Storage of position data in each iteration
Vj = pd.DataFrame(columns=particle_list)  # Storage of velocity data in each iteration
Route_j = pd.DataFrame(columns=particle_list)  # Storage of the routes result
Cost_j = pd.DataFrame(columns=particle_list) # Storage of the costs result
Fitness_j = pd.DataFrame(columns=particle_list) # Storage of the fitness value
Pbest_j = pd.DataFrame(columns=particle_list) # Storage of the Pbest in each iteration
Gbest_i = pd.DataFrame(columns=["Gbest","Fitness Value"]) # Storage of the Gbest in each iteration

# Grouping Dataframe
Result = [Xj, Vj, Route_j, Cost_j, Fitness_j, Pbest_j, Gbest_i]

# Execute PSO algorithm
Result = PSO_exe(Result, Master_df, n, N, d, particle_list, X_min, X_max, V_min, V_max)
