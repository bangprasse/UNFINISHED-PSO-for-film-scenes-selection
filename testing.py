# Import functions from modules
from modules import *

# clearscreen()
# DATA SOURCE INIT AND PREPROCCESING
# -------------------------------------------
# Input datasource to Master_df
AllScn_df = pd.DataFrame(
    pd.read_excel(
        "Datasource/Master Data.xlsx",
        sheet_name="Scenes Data",
    )
)
# # Master_df = Master_df.drop(Master_df.columns[0], axis=1)
# print_df(AllScn_df)

# AllLoc_df = pd.DataFrame(
#     pd.read_excel(
#         "Datasource/Master Data.xlsx",
#         sheet_name="Location Distance",
#     )
# )
# AllLoc_df.set_index("Location", inplace=True)
# print_df(AllLoc_df)

# Scene_df = AllScn_df.copy()

# # Get All Scenes Data
# Scene_name = Scene_df['Scene'].to_list()

# # Calculate Distance of Each Scene
N = 100
# clearing_df(Scene_df)
print("==================== PSO Result Summary ====================")
print(">> Swarm Size (N)    : ", N)
