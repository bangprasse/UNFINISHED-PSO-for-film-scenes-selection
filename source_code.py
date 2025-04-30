# IMPORT PACKAGE AND LIBRARIES
# -----------------------------------
import numpy as np
import pandas as pd
from function_def import *  # Include function_def.py
import tkinter as tk
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# DATA SOURCE INIT
# -----------------------------------
# All Scenes Data
scenes_df = pd.DataFrame(
    pd.read_excel(
        "Master Data.xlsx",
        sheet_name="Scenes Data",
    )
)
print_df(scenes_df)

# Distance Each Scenes
scn_dist_df = pd.DataFrame(
    pd.read_excel(
        "Master Data.xlsx",
        sheet_name="Scenes Distance",
    )
)
print_df(scn_dist_df)

# Distance Each Location
loc_dist_df = pd.DataFrame(
    pd.read_excel(
        "Master Data.xlsx",
        sheet_name="Location Distance",
    )
)
print_df(loc_dist_df)

# Base dataframe that will use here
Base_df = loc_dist_df.copy()


# PARAM INIT
# -----------------------------------
n = 100  # Max Iteration
N = 25  # Swarm size
d = len(Base_df)  # Dimension

# Learning Rates
c1 = 2
c2 = 2

# Position Clamping
Xmin = 0
Xmax = 1

# Velocity Clamping
Vmax = rd.uniform(Xmin, Xmax)
Vmin = -Vmax

# Starter Particle Position
