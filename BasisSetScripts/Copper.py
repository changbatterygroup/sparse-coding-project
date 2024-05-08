import ast
import os
import numpy as np
import pandas as pd
import sklearn

DataDir = "~/OneDrive - Drexel University/Documents - Chang Lab/General/Group/Data/Ultrasound/Layered electrode study/"
def parse_array_string(arr_str):
    arr = ast.literal_eval(arr_str)
    arr = np.array(arr, dtype=float)
    return arr


