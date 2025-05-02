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
c1 = 2  # Learning Rates
c2 = 2  # Learning Rates

# Inertia Weight
# here, i use random inertia weight strategy every iteration
# so, i generate it by function in function_def.py

# Position Clamping
X_min = 0
X_max = 1

# Generate Xj and Initial Position
Xj = initial_swarm_position(N, d, X_min, X_max)
particle_list = Xj.columns.to_list()
print_df(Xj)

# Velocity Clamping
V_max = (rd.uniform(0, 1)) * (X_max - X_min)
V_min = -V_max

# Generate Vj and Initial Velocity
Vj = initial_swarm_velocity(N, d, V_min, V_max)
print_df(Vj)

# Evaluate Initial Route and Its Cost
RouteValue_j = pd.DataFrame(columns = particle_list)
RouteValue_j = evaluate_route(RouteValue_j, Xj, 0, N)
print_df(RouteValue_j)

CostValue_j = pd.DataFrame(columns = particle_list)
CostValue_j = evaluate_cost(CostValue_j, RouteValue_j, 0, Master_df, N)
print_df(CostValue_j)

# Evaluate Fitness value


