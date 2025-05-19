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

cost = 0
for r in range(0, len(Starting_Route)):
    if r == len(Starting_Route) - 1:
        vertex1 = Starting_Route[r]
        vertex2 = Starting_Route[0]
    else:
        vertex1 = Starting_Route[r]
        vertex2 = Starting_Route[r + 1]
    print(vertex1, " -> ", vertex2)
    cost = cost + Cost_df[vertex2][vertex1]
    print(cost)
print(cost)
