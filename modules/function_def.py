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


def initial_swarm_position(swarmsize, dimension, Xmin, Xmax):
    """ """
    X = pd.DataFrame()

    for i in range(0, swarmsize):
        idx = "X" + str(i + 1)
        position = [[round(rd.uniform(Xmin, Xmax), 6) for j in range(0, dimension)]]
        X[idx] = position

    return X


def initial_swarm_velocity(swarmsize, dimension, Vmin, Vmax):
    """"""
    V = pd.DataFrame()

    for i in range(0, swarmsize):
        idx = "V" + str(i + 1)
        velocity = [[round(rd.uniform(Vmin, Vmax), 6) for j in range(0, dimension)]]
        V[idx] = velocity

    return V


def evaluate_route(dataframe, Xj, iteration, swarmsize):
    """"""
    Xj = pd.DataFrame(Xj)
    Routes = pd.DataFrame()

    for i in range(0, swarmsize):
        idx = "X" + str(i + 1)
        unsorted_list = Xj[idx][iteration]
        sorted_list = [
            sorted(range(len(unsorted_list)), key=lambda x: unsorted_list[x])
        ]
        Routes[idx] = sorted_list
    Routes = pd.concat([dataframe, Routes], ignore_index=True)

    return Routes


def evaluate_cost(dataframe, Routes_df, iteration, Source_df, swarmsize):
    """"""
    Routes_df = pd.DataFrame(Routes_df)
    Source_df = pd.DataFrame(Source_df)
    AllCost = pd.DataFrame()
    for i in range(0, swarmsize):
        idx = "X" + str(i + 1)
        route = Routes_df[idx][iteration]
        cost = 0
        for j in range(0, len(route) - 1):
            vertex1 = route[j]
            vertex2 = route[j + 1]
            cost = cost + (Source_df[vertex2][vertex1])
        AllCost[idx] = [cost]
    AllCost = pd.concat([dataframe, AllCost], ignore_index=True)

    return AllCost


def PSO_exe(max_iter: int, swarmsize: int, dimension: int, Xmin, Xmax, Vmin, Vmax):
    """"""
    # Generate new dataframe
    Xj = pd.DataFrame() # Storage for position data in each iteration
    Vj = pd.DataFrame() # Storage for velocity data in each iteration
    Route_j = pd.DataFrame() # Storage for the route result in each iteration 

    for i in range(0, max_iter + 1):
        if i == 0:
            # Generate Initial Position
            Xj = initial_swarm_position(swarmsize, dimension, Xmin, Xmax)

            # Get particle name by the column
            particle_names = Xj.columns.to_list()

            # Generate Initial Velocity
            Vj = initial_swarm_velocity(swarmsize, dimension, Vmin, Vmax)
        # else:
        

        # Evaluate New Route
        

