from modules import *

# DATA SOURCE INIT AND PREPROCCESING
# -------------------------------------------
# 1. Input: Data Source for All Scenes
AllScn_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data.xlsx",  # Datasource filepath
        sheet_name="Scenes Data",
    )
)
AllScn_df = clearing_df(AllScn_df)  # Clearing AllScn_df
AllScn_df.drop(AllScn_df.columns[0], axis=1, inplace=True)  # Delete "No" Col
AllScn_df.set_index("Scene", inplace=True)

# Get the name of the scene
scene_name = AllScn_df.index.to_list()


# 2. Input: Data Source for Distance Between All Locations
LocDis_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data.xlsx",
        sheet_name="Location Distance",
    )
)
LocDis_df.set_index("Location", inplace=True)
Location_names = LocDis_df.index.to_list()


# 3. Input: Data Source for Cost per Scene of Each Talent
CosTal_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data.xlsx",
        sheet_name="Talents Data",
    )
)
CosTal_df.set_index("Character", inplace=True)


# 4. Data Source for Total Talent Cost per Scene
CosTal_Scn_df = pd.DataFrame()
for scene in scene_name:
    # Get talent in scene
    talents = AllScn_df["Cast"][scene]
    talents = talents.split(", ")

    # Calculate Total Cost
    cost = 0
    for talent in talents:
        if talent != "-":
            cost = cost + CosTal_df["Cost Per Scene"][talent]
    CosTal_Scn_df[scene] = [cost]

# -------------------------------------------------------------------------------------------
# Uncomment all in this block if using scene

# # 5. Data Source for Distance Between All Scenes
# ScnDis_df = pd.DataFrame(columns=scene_name)
# for scene_start in scene_name:
#     loc_start = AllScn_df["Location"][scene_start]
#     Scn_temp_df = pd.DataFrame()  # Temporary Storage
#     for scene_dest in scene_name:
#         loc_dest = AllScn_df["Location"][scene_dest]
#         if loc_start == "-" or loc_dest == "-":
#             distance = 0
#         else:
#             distance = LocDis_df[loc_dest][loc_start]
#         Scn_temp_df[scene_dest] = [distance]
#     ScnDis_df = pd.concat([ScnDis_df, Scn_temp_df], ignore_index=True)
# ScnDis_df.index = scene_name


# # 6. Data Source for Fuel Cost Between All Scenes
# Cars = 3
# Dist_per_liter = 8000  # Distance (meter) that Car move per Liter Fuel
# Fuel_Price = 10000  # Fuel Price (Rupiah)
# Fuel_Cost_per_meter = 3 * (Fuel_Price / Dist_per_liter)  # Fuel Cost for Cars per meter

# Fuel_Cost_df = ScnDis_df.copy()
# Fuel_Cost_df = Fuel_Cost_df * Fuel_Cost_per_meter


# # 7. Data Source for Total Cost Between All Scenes
# Cost_df = pd.DataFrame(columns=scene_name)
# for scene_start in scene_name:
#     Cost_temp_df = pd.DataFrame()
#     for scene_dest in scene_name:
#         Fuel_Cost = Fuel_Cost_df[scene_dest][scene_start]
#         Tal_Cost = CosTal_Scn_df[scene_dest][0]
#         Tot_cost = Fuel_Cost + Tal_Cost
#         Cost_temp_df[scene_dest] = [Tot_cost]
#     Cost_df = pd.concat([Cost_df, Cost_temp_df], ignore_index=True)
# Cost_df.index = scene_name
# ------------------------------------------------------------------------
# Uncomment this block if you using location

# 5. Data Source for Talent Cost per Location
# Copy All Scene Data
C_AllScn_df = AllScn_df.copy()
C_AllScn_df.set_index("Location")

CosTal_Loc_df = pd.DataFrame(index=Location_names)
scene_play = []
cost_play = []
# Get Scenes in Each Location and Total Talent Cost in Each Location
for location in Location_names:
    scenes = C_AllScn_df.index[C_AllScn_df["Location"] == location].to_list()
    TotTalCost = 0
    for scene in scenes:
        TotTalCost = TotTalCost + CosTal_Scn_df[scene][0]
    scene_play.append(scenes)
    cost_play.append(TotTalCost)
CosTal_Loc_df["Scenes"] = scene_play
CosTal_Loc_df["Total Talent Cost"] = cost_play


# 6. Data Source for Fuel Cost Between All Locations
Cars = 3
Dist_per_liter = 8000  # Distance (meter) that Car move per Liter Fuel
Fuel_Price = 10000  # Fuel Price (Rupiah)
Fuel_Cost_per_meter = 3 * (Fuel_Price / Dist_per_liter)  # Fuel Cost for Cars per meter

Fuel_Cost_df = LocDis_df.copy()
Fuel_Cost_df = Fuel_Cost_df * Fuel_Cost_per_meter


# 7. Data Source for Total Cost Between All Locations
Cost_df = pd.DataFrame(columns=Location_names)
for loc_start in Location_names:
    Cost_temp_df = pd.DataFrame()
    for loc_dest in Location_names:
        Fuel_Cost = Fuel_Cost_df[loc_dest][loc_start]
        Tal_Cost = CosTal_Loc_df["Total Talent Cost"][loc_dest]
        Tot_cost = Fuel_Cost + Tal_Cost
        Cost_temp_df[loc_dest] = [Tot_cost]
    Cost_df = pd.concat([Cost_df, Cost_temp_df], ignore_index=True)
Cost_df.index = Location_names
