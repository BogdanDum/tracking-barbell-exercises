import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# Load data
df = pd.read_pickle("../../data/interim/removed_outliers_chauvenet_02.pkl")

predictor_columns = list(df.columns[:6])


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2




# Deal with missing values using imputation
df.info()

subset = df[df["set"] == 35]["gyr_y"].plot()

# interpolate the values linearly 
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()
# now there are no missing values anymore



# Compute the set duration
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 47]["acc_y"].plot()
# we can already notice some repetitive paths in the graph for each rep



# Use the 





