import pandas as pd
import numpy as np
import random as rd

N = 10
d = 5
Xj = pd.DataFrame()

# Global dataframe
A = pd.DataFrame(columns=["A", "B"])
print(A)


def add_row_to_A(df, a_val, b_val):
    new_row = pd.DataFrame([{"A": a_val, "B": b_val}])
    df = pd.concat([df, new_row], ignore_index=True)
    return df


# Call and update
A = add_row_to_A(A, 1, 2)
A = add_row_to_A(A, 3, 5)
print(A)
