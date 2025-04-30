# IMPORT PACKAGE AND LIBRARIES
# -----------------------------------
import numpy as np
import pandas as pd
import random as rd
from tabulate import tabulate as tb
import os


# USER DEFINE FUNCTION
# -----------------------------------
# Prettier Tabular Output
def print_df(dataframe):
    print(tb(dataframe, headers="keys", tablefmt="psql"))


# Clear Output Screen
def clearscreen():
    os.system("cls" if os.name == "nt" else "clear")
