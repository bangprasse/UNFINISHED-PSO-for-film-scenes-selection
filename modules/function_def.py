# IMPORT PACKAGE AND LIBRARIES
# -----------------------------------
import numpy as np
import pandas as pd
import random as rd
from tabulate import tabulate as tb
import os


# USER DEFINE FUNCTION
# -----------------------------------
def print_df(df_name: str, df: pd.DataFrame):
    """
    Prettier Tabular Output.

    Args:
        df_name: str                    = Name of The Dataframe
        df: pandas.core.frame.DataFrame = The Dataframe that will be print out.
    """
    print(">> " + df_name)
    print(tb(df, headers="keys", tablefmt="psql"))
    print("")


def clearscreen():
    """
    Clears the terminal screen.

    Uses the appropriate command depending on the operating system:
    - 'cls' for Windows
    - 'clear' for Unix/Linux/Mac
    """
    os.system("cls" if os.name == "nt" else "clear")


def clearing_df(df: pd.DataFrame):
    df = df.copy()

    # Replace all "None", blank, "NaN" value
    df = df.fillna("-")

    return df


def initial_swarm_position(
    X_df: pd.DataFrame, dimension: int, particle_names: list, Xmin: float, Xmax: float
):
    """ """
    # Securing Source Dataframe
    X_df = X_df.copy()

    pos = pd.DataFrame()  # Temporary storage
    for particle in particle_names:  # Generate Initial Positions
        position = [[round((rd.uniform(Xmin, Xmax)), 6) for j in range(0, dimension)]]
        pos[particle] = position

    X_df = pd.concat([X_df, pos], ignore_index=True)

    return X_df


def initial_swarm_velocity(
    V_df: pd.DataFrame, dimension: int, particle_names: list, Vmin: float, Vmax: float
):
    """"""
    # Securing Source Dataframe
    V_df = V_df.copy()

    # adding r1, r2 to initial velocity
    velo = pd.DataFrame()  # Temporary storage
    velo["r1"] = [0]
    velo["r2"] = [0]

    for particle in particle_names:  # Generate Initial Velocities
        velocity = [[round((rd.uniform(Vmin, Vmax)), 6) for j in range(0, dimension)]]
        velo[particle] = velocity

    V_df = pd.concat([V_df, velo], ignore_index=True)

    return V_df


def evaluate_route(
    R_df: pd.DataFrame, X_df: pd.DataFrame, particle_names: list, iteration: int
):
    """"""
    # Securing Source Dataframe
    R_df = R_df.copy()
    X_df = X_df.copy()

    Routes = pd.DataFrame()  # Temporary Storage
    for particle in particle_names:  # Sorting Routes for Each Particle
        position = X_df[particle][iteration]
        sort_route = [sorted(range(len(position)), key=lambda x: position[x])]
        Routes[particle] = sort_route

    R_df = pd.concat([R_df, Routes], ignore_index=True)

    return R_df


def evaluate_cost(
    C_df: pd.DataFrame,
    R_df: pd.DataFrame,
    CDS_df: pd.DataFrame,
    particle_names: list,
    iteration=int,
):
    """"""
    Costs = pd.DataFrame()  # Temporary Storage
    for particle in particle_names:
        cost = 0
        route = R_df[particle][iteration]
        for idx in range(0, len(route) - 1):
            vertex1 = route[idx]
            vertex2 = route[idx + 1]
            cost = cost + (CDS_df[vertex2][vertex1])
        Costs[particle] = [cost]

    C_df = pd.concat([C_df, Costs], ignore_index=True)

    return C_df


def evaluate_fitness(
    F_df: pd.DataFrame, C_df: pd.DataFrame, particle_names: list, iteration=int
):
    Fitness = pd.DataFrame()  # Temporary Storage
    for particle in particle_names:
        cost = C_df[particle][iteration]
        fit_val = round(1 / cost, 6)
        Fitness[particle] = [fit_val]

    F_df = pd.concat([F_df, Fitness], ignore_index=True)

    return F_df


def calc_fit_pos(pos: list, CDS_df: pd.DataFrame):
    # Sorting Route
    route = sorted(range(len(pos)), key=lambda x: pos[x])

    # Calculate Fitness Value
    cost = 0
    for r in range(0, len(route) - 1):
        vertex1 = route[r]
        vertex2 = route[r + 1]
        cost = cost + (CDS_df[vertex2][vertex1])
    fitval = round(1 / cost, 6)

    return fitval


def evaluate_pbest(
    P_df: pd.DataFrame,
    F_df: pd.DataFrame,
    X_df: pd.DataFrame,
    R_df: pd.DataFrame,
    CDS_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
):
    # Get Fitness Value in Now Iteration
    fit_val_now = F_df.iloc[iteration]

    Pc_df = pd.DataFrame()  # Temporary Storage
    for particle in particle_names:
        if iteration == 0:
            # Save Now Position as Pbest
            Pc_df[particle] = [X_df[particle][0]]
        else:
            # Call The Pbest and its fitness value in iteration-1 of the particle
            Pb_part_bfr = P_df[particle][iteration - 1]
            Pb_fit_val = calc_fit_pos(Pb_part_bfr, CDS_df)

            # Save the position with biggest Fitness Value as Pbest
            if fit_val_now[particle] >= Pb_fit_val:
                Pc_df[particle] = [X_df[particle][iteration]]
            else:
                Pc_df[particle] = [Pb_part_bfr]

    P_df = pd.concat([P_df, Pc_df], ignore_index=True)

    return P_df


def evaluate_gbest(
    G_df: pd.DataFrame,
    P_df: pd.DataFrame,
    CDS_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
):
    # Call All of Now-Iteration Pbest
    P_best = P_df.iloc[iteration]

    # Find The Gbest Candidat in Now-Iteration
    Gpos_candidat = []
    Gval_candidat = 0
    for particle in particle_names:
        Pb_particle = P_best[particle]
        Pb_val = calc_fit_pos(Pb_particle, CDS_df)

        if Pb_val > Gval_candidat:
            Gpos_candidat = Pb_particle
            Gval_candidat = Pb_val

    Gc_df = pd.DataFrame()  # Temporary Storage
    if iteration == 0:
        Gc_df["Gbest"] = [Gpos_candidat]
        Gc_df["Fitness Value"] = [Gval_candidat]
    else:
        # Call The Gbest of Iteration-1
        G_bfr_val = G_df["Fitness Value"][iteration - 1]

        if Gval_candidat >= G_bfr_val:
            Gc_df["Gbest"] = [Gpos_candidat]
            Gc_df["Fitness Value"] = [Gval_candidat]
        else:
            Gc_df["Gbest"] = [G_df["Gbest"][iteration - 1]]
            Gc_df["Fitness Value"] = [G_bfr_val]

    G_df = pd.concat([G_df, Gc_df], ignore_index=True)

    return G_df


def generate_inertia_weight(IW_df: pd.DataFrame, iteration: int):
    # Generate now-iteration's inertia weight
    inert_w = 0.5 + (round(rd.uniform(0, 1) / 2, 6))

    # Saving to temporary storage
    IWc_df = pd.DataFrame({"Inertia Weight": [inert_w]})

    # Add to the Inertia Weight Dataframe as a new row
    IW_df = pd.concat([IW_df, IWc_df], ignore_index=True)

    return IW_df


def update_velocity(
    V_df: pd.DataFrame,
    X_df: pd.DataFrame,
    IW_df: pd.DataFrame,
    P_df: pd.DataFrame,
    G_df: pd.DataFrame,
    particle_names: list,
    c1: float,
    c2: float,
    iteration: int,
    Vmin: float,
    Vmax: float,
):
    # Call Inertia Weight Now-Iteration
    inert_w = IW_df["Inertia Weight"][iteration]

    # Get the Gbest at the iteration-1
    G_best = np.array(G_df["Gbest"][iteration - 1])

    # Generate value of r1 and r2
    r1 = round(rd.uniform(0, 1), 6)
    r2 = round(rd.uniform(0, 1), 6)

    # adding r1, r2 to initial velocity
    velo = pd.DataFrame()  # Temporary storage
    velo["r1"] = [r1]
    velo["r2"] = [r2]

    # Update the Velocity of each particle
    for particle in particle_names:
        # Get the particle's position at the iteration-1
        X_bfr = np.array(X_df[particle][iteration - 1])

        # Get the particle's velocity at the iteration-1
        V_bfr = np.array(V_df[particle][iteration - 1])

        # Get the particle's Pbest at the iteration-1
        P_bfr = np.array(P_df[particle][iteration - 1])

        # Calculate New Velocity
        V_now = (
            (inert_w * V_bfr)
            + (c1 * r1 * (P_bfr - X_bfr))
            + (c2 * r2 * (G_best - X_bfr))
        )
        V_now = np.round(V_now, 6)

        # check velocity clamping
        for idx in range(0, len(V_now)):
            if V_now[idx] < Vmin:
                V_now[idx] = Vmin
            elif V_now[idx] > Vmax:
                V_now[idx] = Vmax

        # # Convert to a list datatype
        # V_now = V_now.to_list()

        velo[particle] = [V_now]

    V_df = pd.concat([V_df, velo], ignore_index=True)

    return V_df


def update_position(
    X_df: pd.DataFrame,
    V_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    Xmin: float,
    Xmax: float,
):
    pos = pd.DataFrame()  # A Temporary Storage
    for particle in particle_names:
        # Get the particle's position at the iteration-1
        X_bfr = np.array(X_df[particle][iteration - 1])

        # Get the particle's velocity at the now-iteration
        V_now = np.array(V_df[particle][iteration])

        # Calculate New Position
        X_now = X_bfr + V_now

        # Checking Position Clamping
        for idx in range(0, len(X_now)):
            if X_now[idx] < Xmin:
                X_now[idx] = Xmin
            elif X_now[idx] > Xmax:
                X_now[idx] = Xmax

        # Round to 6 number digits after the comma
        X_now = np.round(X_now, 6)

        # Convert to a list datatype
        X_now = X_now.tolist()

        pos[particle] = [X_now]

    X_df = pd.concat([X_df, pos], ignore_index=True)

    return X_df


def PSO_exe(
    Storage: list,
    CDS_df: pd.DataFrame,
    max_iter: int,
    swarmsize: int,
    dimension: int,
    particle_names: list,
    c1: float,
    c2: float,
    Xmin: float,
    Xmax: float,
    Vmin: float,
    Vmax: float,
):
    """"""
    # Ungroup Dataframe
    X_df = Storage[0]  # Storage of Position Data
    V_df = Storage[1]  # Storage of Velocity Data
    R_df = Storage[2]  # Storage of The Route
    C_df = Storage[3]  # Storage of The Cost
    F_df = Storage[4]  # Storage of The Fitness Value
    P_df = Storage[5]  # Storage of The Pbest
    G_df = Storage[6]  # Storage of The Gbest
    IW_df = Storage[7]  # Storage of The Inertia Weight

    for i in range(0, max_iter):  # Start Iteration
        if i == 0:
            # Generate Initial Position
            X_df = initial_swarm_position(X_df, dimension, particle_names, Xmin, Xmax)

            # Generate Initial Velocity
            V_df = initial_swarm_velocity(V_df, dimension, particle_names, Vmin, Vmax)
        else:
            # Generate Inertia Weight
            IW_df = generate_inertia_weight(IW_df, i)

            # Updating Velocity
            V_df = update_velocity(
                V_df, X_df, IW_df, P_df, G_df, particle_names, c1, c2, i, Vmin, Vmax
            )

            # Updating Position
            X_df = update_position(X_df, V_df, particle_names, i, Xmin, Xmax)

        # Constructing The Route
        R_df = evaluate_route(R_df, X_df, particle_names, i)

        # Calculating The Cost of The Route
        C_df = evaluate_cost(C_df, R_df, CDS_df, particle_names, i)

        # Evaluate The Fitness Value of The Route
        F_df = evaluate_fitness(F_df, C_df, particle_names, i)

        # Evaluate P_best
        P_df = evaluate_pbest(P_df, F_df, X_df, R_df, CDS_df, particle_names, i)

        # Evaluate G_best
        G_df = evaluate_gbest(G_df, P_df, CDS_df, particle_names, i)

    # Change all the columns name of each df like the df name
    Result = [X_df, V_df, R_df, C_df, F_df, P_df, G_df, IW_df]

    return Result
