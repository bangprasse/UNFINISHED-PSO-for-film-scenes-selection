# IMPORT PACKAGE AND LIBRARIES
# -----------------------------------
import numpy as np
import pandas as pd
import random as rd
from tabulate import tabulate as tb
import os


# USER DEFINE FUNCTION
# -----------------------------------
def print_df(dataframe):
    """
    Prettier Tabular Output.

    Args:
        dataframe (pandas.core.frame.DataFrame): The Dataframe that will be print out.
    """
    print(tb(dataframe, headers="keys", tablefmt="psql"))


def clearscreen():
    """
    Clears the terminal screen.

    Uses the appropriate command depending on the operating system:
    - 'cls' for Windows
    - 'clear' for Unix/Linux/Mac
    """
    os.system("cls" if os.name == "nt" else "clear")


def initial_swarm_positions(N, d, Xmin, Xmax):
    """ """
    X = pd.DataFrame()
    for i in range(0, N):
        idx = "X" + str(i + 1)
        position = [[rd.uniform(Xmin, Xmax) for j in range(0, d)]]
        X[idx] = position

    return X
