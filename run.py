# IMPORT PACKAGES, /modules, AND LIBRARIES
# -------------------------------------------
# Import functions from modules
from modules import *


# DATA SOURCE INIT
# -------------------------------------------
# Input datasource to Master_df
Master_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data.xlsx",
        sheet_name="Location Distance",
    )
)


# PSO INITIALIZATION
# -------------------------------------------
# Base Parameter
n = 100  # Max Iteration
N = 25  # Swarm Size
d = len(Master_df)  # Dimention Size
c1 = 2  # Learning Rates
c2 = 2  # Learning Rates

# Inertia Weight
# here, i use random inertia weight strategy every iteration
# so, i generate it by function in function_def.py

# Position Clamping
X_min = 0
X_max = 1

# Generate Xj and Initial Position
Xj = initial_swarm_positions(N, d, X_min, X_max)
print_df(Xj)
