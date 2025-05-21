# IMPORT PACKAGE AND LIBRARIES
# -----------------------------------
import numpy as np
import pandas as pd
import random as rd
from tabulate import tabulate as tb
import os
from openpyxl import load_workbook
from openpyxl.styles import Border, Side


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


def folder_maker(folder_name: str):
    base_direct = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(base_direct, "Output", folder_name)

    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def summary_n_resource_export(
    resource: list,
    folder_path: str,
    file_name: str,
):
    # Ungroup Resource
    summary = resource[0]
    scenes = resource[1]
    locations = resource[2]
    talents = resource[3]
    talent_cost = resource[4]
    scenes_distance = resource[5]
    fuel_cost = resource[6]
    scenes_cost = resource[7]
    norm_cost = resource[8]

    # Define folder and file path
    file_name = "01_Summary Result and Resource.xlsx"
    folder_path = os.path.join(folder_path, file_name)

    # Exporting Dataframe
    with pd.ExcelWriter(folder_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary Result", index=True, header=False)
        scenes.to_excel(writer, sheet_name="Scene Data", index=True)
        locations.to_excel(writer, sheet_name="Location Data", index=True)
        talents.to_excel(writer, sheet_name="Talent Data", index=True)
        talent_cost.to_excel(writer, sheet_name="Talent Cost Data", index=True)
        scenes_distance.to_excel(writer, sheet_name="Scene Distance Data", index=True)
        fuel_cost.to_excel(writer, sheet_name="Fuel Cost Data", index=True)
        scenes_cost.to_excel(writer, sheet_name="Scene Cost Data", index=True)
        norm_cost.to_excel(writer, sheet_name="Normalized Scene Cost Data", index=True)


def calculation_export(
    result: pd.DataFrame,
    folder_path: str,
    file_name: str,
    condition: str,
    sheetname: str,
    head: bool,
    row: bool,
):
    # Define folder and file path
    file_name = file_name + ".xlsx"
    folder_path = os.path.join(folder_path, file_name)

    if condition == "new":
        result.to_excel(folder_path, sheet_name=sheetname, header=head, index=row)
    else:
        with pd.ExcelWriter(folder_path, mode="a", engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name=sheetname, header=head, index=row)


def initial_swarm_position(
    X_df: pd.DataFrame,
    dimension: int,
    particle_names: list,
    Xmin: float,
    Xmax: float,
    new_scn_idx: list,
    folder_path: str,
):
    """ """
    # Securing Source Dataframe
    X_df = X_df.copy()
    Ex_pos = pd.DataFrame(columns=new_scn_idx)

    pos = {}  # Temporary storage
    for particle in particle_names:  # Generate Initial Positions
        position = [round((rd.uniform(Xmin, Xmax)), 7) for j in range(0, dimension)]
        pos[particle] = [position]
        idx = particle.replace("Prt", "X")
        Ex_pos.loc[idx] = position

    calculation_export(
        Ex_pos, folder_path, "02_Position", "new", "Iteration 0", True, True
    )
    pos = pd.DataFrame.from_dict(pos).copy()
    X_df = pd.concat([X_df, pos], ignore_index=True)

    return X_df


def initial_swarm_velocity(
    V_df: pd.DataFrame,
    dimension: int,
    particle_names: list,
    Vmin: float,
    Vmax: float,
    new_scn_idx: list,
    folder_path: str,
):
    """"""
    scena = ["r1", "r2"]
    for scn in new_scn_idx:
        scena.append(scn)
    Ex_V = pd.DataFrame(columns=scena)

    # adding r1, r2 to initial velocity
    velo_dict = {}  # Temporary storage
    velo_dict["r1"] = [0]
    velo_dict["r2"] = [0]

    for particle in particle_names:  # Generate Initial Velocities
        velocity = [round((rd.uniform(Vmin, Vmax)), 7) for j in range(0, dimension)]
        velo_dict[particle] = [velocity]

        v_velo = [0, 0]
        for vel in velocity:
            v_velo.append(vel)

        v_idx = particle.replace("Prt", "V")
        Ex_V.loc[v_idx] = v_velo

    calculation_export(
        Ex_V, folder_path, "04_Velocity", "new", "Iteration 0", True, True
    )
    velo = pd.DataFrame.from_dict(velo_dict).copy()
    V_df = pd.concat([V_df, velo], ignore_index=True).copy()

    return V_df


def evaluate_route(
    R_df: pd.DataFrame,
    X_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    start_idx: int,
    scn_idx: list,
):
    """"""
    # Securing Source Dataframe
    R_df = R_df.copy()
    X_df = X_df.copy()

    Routes = {}  # Temporary Storage
    for particle in particle_names:  # Sorting Routes for Each Particle
        position = X_df[particle][iteration]
        route = dict(zip(scn_idx, position))
        sort_route = sorted(route, key=route.get)

        # Adding the starter scene as the start and ending point
        sort_route.insert(0, start_idx)
        sort_route.append(start_idx)
        Routes[particle] = [sort_route]

    Routes = pd.DataFrame.from_dict(Routes).copy()
    R_df = pd.concat([R_df, Routes], ignore_index=True)

    return R_df


def evaluate_cost(
    C_df: pd.DataFrame,
    R_df: pd.DataFrame,
    CDS_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    folder_path: str,
):
    """"""
    Ex_RnC = pd.DataFrame(columns=["Route", "Cost"])

    Costs = {}  # Temporary Storage
    for particle in particle_names:
        cost = 0
        route = R_df[particle][iteration]
        for idx in range(0, len(route) - 1):
            vertex1 = route[idx]
            vertex2 = route[idx + 1]
            cost = cost + (CDS_df[vertex2][vertex1])
        Costs[particle] = [cost]

        rnc_val = [route, cost]
        rnc_idx = particle.replace("Prt", "X")
        Ex_RnC.loc[rnc_idx] = rnc_val

    if iteration == 0:
        calculation_export(
            Ex_RnC, folder_path, "05_Route and Cost", "new", "Iteration 0", True, True
        )
    else:
        calculation_export(
            Ex_RnC,
            folder_path,
            "05_Route and Cost",
            "Add",
            "Iteration " + str(iteration),
            True,
            True,
        )
    Costs = pd.DataFrame.from_dict(Costs).copy()
    C_df = pd.concat([C_df, Costs], ignore_index=True)

    return C_df


def evaluate_fitness(
    F_df: pd.DataFrame,
    C_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    max_iter: int,
    folder_path: str,
):
    Fitness = {}  # Temporary Storage
    f_idx = []
    for particle in particle_names:
        cost = C_df[particle][iteration]
        fit_val = round(1 / cost, 7)
        Fitness[particle] = [fit_val]
        f_idx.append(particle.replace("Prt", "X"))
    Fitness = pd.DataFrame.from_dict(Fitness).copy()
    F_df = pd.concat([F_df, Fitness], ignore_index=True)

    if iteration == max_iter:
        Ex_F = F_df.copy()
        Ex_F.columns = f_idx
        f_i = [i for i in range(0, max_iter + 1)]
        Ex_F["Iteration"] = f_i
        Ex_F = Ex_F.set_index("Iteration")
        calculation_export(
            Ex_F, folder_path, "06_Fitness Value", "new", "Fitness Value", True, True
        )

    return F_df


def calc_fit_pos(pos: list, CDS_df: pd.DataFrame, start_idx: int, scn_idx: list):
    route = dict(zip(scn_idx, pos))
    sort_route = sorted(route, key=route.get)

    # Adding the starter scene as the start and ending point
    sort_route.insert(0, start_idx)
    sort_route.append(start_idx)

    # Calculate Fitness Value
    cost = 0
    for r in range(0, len(sort_route) - 1):
        vertex1 = sort_route[r]
        vertex2 = sort_route[r + 1]
        cost = cost + (CDS_df[vertex2][vertex1])
    fitval = round(1 / cost, 7)

    return fitval


def evaluate_pbest(
    P_df: pd.DataFrame,
    F_df: pd.DataFrame,
    X_df: pd.DataFrame,
    R_df: pd.DataFrame,
    CDS_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    start_idx: int,
    scn_idx: list,
    max_iter: int,
    folder_path: str,
):
    # Get Fitness Value in Now Iteration
    fit_val_now = F_df.iloc[iteration]
    Ex_col = ["Fitness Value"]
    for scn in scn_idx:
        Ex_col.append(scn)
    Ex_P = pd.DataFrame(columns=Ex_col)

    Pc_df = {}  # Temporary Storage
    for particle in particle_names:
        if iteration == 0:
            # Save Now Position as Pbest
            Pc_df[particle] = [X_df[particle][0]]

            p_list = [F_df[particle][0]]
            for pos_val in X_df[particle][0]:
                p_list.append(pos_val)
        else:
            # Call The Pbest and its fitness value in iteration-1 of the particle
            Pb_part_bfr = P_df[particle][iteration - 1]
            Pb_fit_val = calc_fit_pos(Pb_part_bfr, CDS_df, start_idx, scn_idx)

            # Save the position with biggest Fitness Value as Pbest
            if fit_val_now[particle] >= Pb_fit_val:
                Pc_df[particle] = [X_df[particle][iteration]]

                p_list = [fit_val_now[particle]]
                for pos_val in X_df[particle][iteration]:
                    p_list.append(pos_val)
            else:
                Pc_df[particle] = [Pb_part_bfr]

                p_list = [Pb_fit_val]
                for pos_val in Pb_part_bfr:
                    p_list.append(pos_val)

        p_part = particle.replace("Prt", "Pbest")
        Ex_P.loc[p_part] = p_list

    if iteration == 0:
        calculation_export(
            Ex_P, folder_path, "07_Pbest", "new", "Iteration 0", True, True
        )
    else:
        calculation_export(
            Ex_P,
            folder_path,
            "07_Pbest",
            "add",
            "Iteration " + str(iteration),
            True,
            True,
        )

    Pc_df = pd.DataFrame.from_dict(Pc_df).copy()
    P_df = pd.concat([P_df, Pc_df], ignore_index=True)

    return P_df


def evaluate_gbest(
    G_df: pd.DataFrame,
    P_df: pd.DataFrame,
    CDS_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    start_idx: int,
    scn_idx: list,
    folder_path: str,
    max_iter: int,
):
    # Call All of Now-Iteration Pbest
    P_best = P_df.iloc[iteration]

    # Find The Gbest Candidat in Now-Iteration
    Gpos_candidat = []
    Gval_candidat = 0
    for particle in particle_names:
        Pb_particle = P_best[particle]
        Pb_val = calc_fit_pos(Pb_particle, CDS_df, start_idx, scn_idx)

        if Pb_val > Gval_candidat:
            Gpos_candidat = Pb_particle
            Gval_candidat = Pb_val

    Gc_df = {}  # Temporary Storage
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

    Gc_df = pd.DataFrame.from_dict(Gc_df).copy()
    G_df = pd.concat([G_df, Gc_df], ignore_index=True)
    if iteration == max_iter:
        Ex_G = pd.DataFrame()
        Ex_idx = [i for i in range(0, max_iter + 1)]
        Ex_G["Iteration"] = Ex_idx
        Ex_G["Fitness Value"] = G_df["Fitness Value"]
        Ex_G["Gbest"] = G_df["Gbest"]
        Ex_G = Ex_G.set_index("Iteration")
        calculation_export(Ex_G, folder_path, "08_Gbest", "new", "Gbest", True, True)

    return G_df


def generate_inertia_weight(
    IW_df: pd.DataFrame, iteration: int, max_iter: int, folder_path: str
):
    # Generate now-iteration's inertia weight
    inert_w = 0.5 + (round(rd.uniform(0, 1) / 2, 7))

    # w_max = 0.9
    # w_min = 0.1
    # inert_w = round(w_max - (iteration * ((w_max - w_min) / max_iter)), 7)

    # Saving to temporary storage
    IWc_df = pd.DataFrame({"Inertia Weight": [inert_w]})

    # Add to the Inertia Weight Dataframe as a new row
    IW_df = pd.concat([IW_df, IWc_df], ignore_index=True)

    if iteration == max_iter:
        iw = IW_df.copy()
        calculation_export(
            iw,
            folder_path,
            "03_Inertia Weight",
            "new",
            "Inertia Weight Value",
            True,
            True,
        )

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
    new_scn_idx: list,
    folder_path: str,
):
    scena = ["r1", "r2"]
    for scn in new_scn_idx:
        scena.append(scn)
    Ex_V = pd.DataFrame(columns=scena)

    # Call Inertia Weight Now-Iteration
    inert_w = IW_df["Inertia Weight"][iteration]

    # Get the Gbest at the iteration-1
    G_best = np.array(G_df["Gbest"][iteration - 1])

    # Generate value of r1 and r2
    r1 = round(rd.uniform(0, 1), 7)
    r2 = round(rd.uniform(0, 1), 7)

    # adding r1, r2 to initial velocity
    velo = {}  # Temporary storage
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
        V_now = np.round(V_now, 7)

        # check velocity clamping
        for idx in range(0, len(V_now)):
            if V_now[idx] < Vmin:
                V_now[idx] = Vmin
            elif V_now[idx] > Vmax:
                V_now[idx] = Vmax
        velo[particle] = [V_now]

        v_velo = [r1, r2]
        for v_vel in V_now:
            v_velo.append(v_vel)
        v_idx = particle.replace("Prt", "V")
        Ex_V.loc[v_idx] = v_velo

    calculation_export(
        Ex_V,
        folder_path,
        "04_Velocity",
        "Add",
        "Iteration " + str(iteration),
        True,
        True,
    )
    velo = pd.DataFrame.from_dict(velo).copy()
    V_df = pd.concat([V_df, velo], ignore_index=True).copy()

    return V_df


def update_position(
    X_df: pd.DataFrame,
    V_df: pd.DataFrame,
    particle_names: list,
    iteration: int,
    Xmin: float,
    Xmax: float,
    new_scn_idx: list,
    folder_path: str,
):
    Ex_pos = pd.DataFrame(columns=new_scn_idx)

    pos = {}  # A Temporary Storage
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
        X_now = np.round(X_now, 7)

        # Convert to a list datatype
        X_now = X_now.tolist()
        pos[particle] = [X_now]

        idx = particle.replace("Prt", "X")
        Ex_pos.loc[idx] = X_now

    calculation_export(
        Ex_pos,
        folder_path,
        "02_Position",
        "Add",
        "Iteration " + str(iteration),
        True,
        True,
    )
    pos = pd.DataFrame.from_dict(pos).copy()
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
    starting_scene: int | str,
    scene_list: list,
    folder_path: str,
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

    # Get the index of the starting scene from the scene name list
    starting_idx = scene_list.index(starting_scene)

    # Excluding starting scene from pso calculation and scene_list
    new_scn_idx = [idx for idx in range(0, len(scene_list))]
    new_scn_idx.remove(starting_idx)

    for i in range(0, max_iter + 1):
        # Start Iteration, adding +1 because 0 was pso init
        if i == 0:
            # Generate Initial Position
            X_df = initial_swarm_position(
                X_df, dimension, particle_names, Xmin, Xmax, new_scn_idx, folder_path
            )

            # Generate Initial Velocity
            V_df = initial_swarm_velocity(
                V_df, dimension, particle_names, Vmin, Vmax, new_scn_idx, folder_path
            )
        else:
            # Generate Inertia Weight
            IW_df = generate_inertia_weight(IW_df, i, max_iter, folder_path)

            # Updating Velocity
            V_df = update_velocity(
                V_df,
                X_df,
                IW_df,
                P_df,
                G_df,
                particle_names,
                c1,
                c2,
                i,
                Vmin,
                Vmax,
                new_scn_idx,
                folder_path,
            )

            # Updating Position
            X_df = update_position(
                X_df, V_df, particle_names, i, Xmin, Xmax, new_scn_idx, folder_path
            )

        # Constructing The Route
        R_df = evaluate_route(R_df, X_df, particle_names, i, starting_idx, new_scn_idx)

        # Calculating The Cost of The Route
        C_df = evaluate_cost(C_df, R_df, CDS_df, particle_names, i, folder_path)

        # Evaluate The Fitness Value of The Route
        F_df = evaluate_fitness(F_df, C_df, particle_names, i, max_iter, folder_path)

        # Evaluate P_best
        P_df = evaluate_pbest(
            P_df,
            F_df,
            X_df,
            R_df,
            CDS_df,
            particle_names,
            i,
            starting_idx,
            new_scn_idx,
            max_iter,
            folder_path,
        )

        # Evaluate G_best
        G_df = evaluate_gbest(
            G_df, P_df, CDS_df, particle_names, i, starting_idx, new_scn_idx, folder_path, max_iter
        )

    # Change all the columns name of each df like the df name
    Result = [X_df, V_df, R_df, C_df, F_df, P_df, G_df, IW_df]

    return Result


def optim_route(route_pos: list, scene_list: list, start_idx: int, scn_idx: list):
    # Sorting Route
    route = dict(zip(scn_idx, route_pos))
    sort_route = sorted(route, key=route.get)

    # Adding the starter scene as the start and ending point
    sort_route.insert(0, start_idx)
    sort_route.append(start_idx)

    route_result = []
    for scene in sort_route:
        route_result.append(scene_list[scene])

    return route_result


def optim_cost(
    Route: list,
    CDS_df: pd.DataFrame,
    max_value: float,
    min_value: float,
):
    cost = 0
    for r in range(0, len(Route) - 1):
        vertex1 = Route[r]
        vertex2 = Route[r + 1]
        cost = cost + CDS_df[vertex2][vertex1]

    return cost


def print_output(
    session_name: str,
    Storage: list,
    CDS_df: pd.DataFrame,
    Cost_df: pd.DataFrame,
    N: int,
    n: int,
    c1: float,
    c2: float,
    Vmax: float,
    Vmin: float,
    scene_list: list,
    max_value: float,
    min_value: float,
    starting_scene: int | str,
):
    # Ungroup Dataframe
    X_df = Storage[0]  # Storage of Position Data
    V_df = Storage[1]  # Storage of Velocity Data
    R_df = Storage[2]  # Storage of The Route
    C_df = Storage[3]  # Storage of The Cost
    F_df = Storage[4]  # Storage of The Fitness Value
    P_df = Storage[5]  # Storage of The Pbest
    G_df = Storage[6]  # Storage of The Gbest
    IW_df = Storage[7]  # Storage of The Inertia Weight

    # Printing All Dataframe Output
    print_df("Output: Dataframe for Position", X_df)
    print_df("Output: Dataframe for Inertia Weight", IW_df)
    print_df("Output: Dataframe for Velocity", V_df)
    print_df("Output: Dataframe for Route", R_df)
    print_df("Output: Dataframe for Cost", C_df)
    print_df("Output: Dataframe for Fitness Value", F_df)
    print_df("Output: Dataframe for Pbest", P_df)
    print_df("Output: Dataframe for Gbest", G_df)

    # Get the index of the starting scene from the scene name list
    starting_idx = scene_list.index(starting_scene)

    # Excluding starting scene from pso calculation and scene_list
    new_scn_idx = [idx for idx in range(0, len(scene_list))]
    new_scn_idx.remove(starting_idx)

    Optimum_Route = optim_route(G_df["Gbest"][n], scene_list, starting_idx, new_scn_idx)
    Optimum_Cost = optim_cost(Optimum_Route, Cost_df, max_value, min_value)
    Optimum_Fitness = round(G_df["Fitness Value"][n], 7)

    # Printing Summary Output
    print("")
    print("==================== PSO Result Summary ====================")
    print(">> Swarm Size (N)    : ", N)
    print(">> Max Iteration (n) : ", n)
    print(">> Cognitive (c1)    : ", c1)
    print(">> Social (c2)       : ", c2)
    print(">> Velocity Max      : ", Vmax)
    print(">> Velocity Min      : ", Vmin)
    print(">> Inertia Weight    : Random Inertia Weight Strategy")
    print("")
    print(">> Best Route        : ", Optimum_Route)
    print(">> Cost              : Rp", Optimum_Cost)

    # Make summary dataframe
    summary = {}
    summary["Calcuation Session"] = [session_name]
    summary["Swarm Size (N)"] = [N]
    summary["Max Iteration (n)"] = [n]
    summary["Cognitive (c1)"] = [c1]
    summary["Social (c2)"] = [c2]
    summary["Velocity Maxmimum (Vmax)"] = [Vmax]
    summary["Velocity Minimum (Vmin)"] = [Vmin]
    summary["Inertia Weight Strategy"] = ["Random Inertia Weight Strategy"]
    summary[" "] = [" "]
    summary["Best Route"] = [Optimum_Route]
    summary["Route Cost"] = ["Rp" + str(Optimum_Cost)]
    summary["Route Fitness Value"] = [str(Optimum_Fitness)]
    summary[" "] = [" "]

    summary = pd.DataFrame.from_dict(summary, orient="index").copy()

    return summary


def Optimality_comparison(
    summary: pd.DataFrame,
    start_route: list,
    Cost_df: pd.DataFrame,
    n: int,
):
    cost = 0
    for r in range(0, len(start_route) - 1):
        vertex1 = start_route[r]
        vertex2 = start_route[r + 1]
        cost_add = Cost_df[vertex2][vertex1]
        cost = round(cost + cost_add, 7)

        # Adding Cost back to the first place
    start_point = start_route[0]
    end_point = start_route[-1]
    cost = cost + Cost_df[end_point][start_point]

    print("--- Optimality Comparison ---")
    print(">> Original Route       : ", start_route)
    print(">> Original Cost        : Rp", cost)

    add_summary = {}
    add_summary["Original Route"] = [start_route]
    add_summary["Original Cost"] = ["Rp" + str(cost)]
    add_summary = pd.DataFrame.from_dict(add_summary, orient="index").copy()

    summary = pd.concat([summary, add_summary], ignore_index=False)

    return summary
