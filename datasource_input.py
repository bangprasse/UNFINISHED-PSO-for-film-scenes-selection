from modules import *

# DATA SOURCE INIT AND PREPROCCESING
# -------------------------------------------
# 1. Input: Data Source for All Scenes
Scn_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data - Copy.xlsx",  # Datasource filepath
        sheet_name="Scenes Data",
    )
)
Scn_df = clearing_df(Scn_df)  # Clearing AllScn_df
Scn_df.drop(Scn_df.columns[0], axis=1, inplace=True)  # Delete "No" Col
Scn_df.set_index("Scene", inplace=True)

# Get the name of all scenes
scene_name = Scn_df.index.to_list()


# 2. Input: Data Source for Distance Between All Locations
Loc_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data - Copy.xlsx",
        sheet_name="Location Distance",
    )
)
Loc_df.set_index("Location", inplace=True)

# Get the name of all locations
Location_names = Loc_df.index.to_list()


# 3. Input: Data Source for Cost per Scene of Each Talent
Tal_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data - Copy.xlsx",
        sheet_name="Talents Data",
    )
)
Tal_df.set_index("Character", inplace=True)

# Get the name of all talent
Talent_names = Tal_df.index.to_list()


# 4. Data Source for Total Talent Cost per Scene
Tal_Cos_df = pd.DataFrame()

for scene in scene_name:
    # Get all talent name in scene
    talents = Scn_df["Cast"][scene]
    talents = talents.split(", ")

    # Calculate Total Talent Cost per scene
    cost = 0
    for talent in talents:
        if talent != "-":  # if no talent it will skip
            cost = cost + Tal_df["Cost Per Scene"][talent]
    Tal_Cos_df[scene] = [cost]


# 5. Data Source for Distance Between All Scenes
Scn_Dis_df = pd.DataFrame(columns=scene_name)

for scene_start in scene_name:
    # Get the name of starting scene location
    loc_start = Scn_df["Location"][scene_start]

    Scn_temp_df = pd.DataFrame()  # Temporary Storage
    for scene_dest in scene_name:
        # Get the name of destionation scene location
        loc_dest = Scn_df["Location"][scene_dest]

        # Get the distance value from starting scene to destination scene
        distance = Loc_df[loc_dest][loc_start]

        Scn_temp_df[scene_dest] = [distance]
    Scn_Dis_df = pd.concat([Scn_Dis_df, Scn_temp_df], ignore_index=True)
Scn_Dis_df.index = scene_name


# 6. Data Source for Fuel Cost Between All Scenes
Cars = 3
Dist_per_liter = 8000  # Distance (meter) that Car move per Liter Fuel
Fuel_Price = 10000  # Fuel Price (Rupiah)
Fuel_Cost_per_meter = 3 * (Fuel_Price / Dist_per_liter)  # Fuel Cost for Cars per meter

Fuel_Cost_df = Scn_Dis_df.copy()
Fuel_Cost_df = Fuel_Cost_df * Fuel_Cost_per_meter
Fuel_Cost_df = Fuel_Cost_df.round(6)


# 7. Data Source for Total Cost Between All Scenes
Cost_df = pd.DataFrame(columns=scene_name)

for scene_start in scene_name:
    Cost_temp_df = pd.DataFrame()  # Temporary Storage
    for scene_dest in scene_name:
        # Get Fuel Cost from starting scene to destination scene
        Fuel_Cost = Fuel_Cost_df[scene_dest][scene_start]

        # Get Talent Cost in destination scene
        Tal_Cost = Tal_Cos_df[scene_dest][0]

        # Calculate Total Cost
        Tot_cost = round((Fuel_Cost + Tal_Cost), 6)
        Cost_temp_df[scene_dest] = [Tot_cost]
    Cost_df = pd.concat([Cost_df, Cost_temp_df], ignore_index=True)
Cost_df.index = scene_name

# Normalization Total Cost value between all scenes
max_val = Cost_df.values.max()
min_val = Cost_df.values.min()
norm_Cost_df = (Cost_df - min_val) / (max_val - min_val)

# Set index to be starting scene name
norm_Cost_df.index = scene_name

# All Resources
Resources = [
    Scn_df,
    Loc_df,
    Tal_df,
    Tal_Cos_df,
    Scn_Dis_df,
    Fuel_Cost_df,
    Cost_df,
    norm_Cost_df
]
