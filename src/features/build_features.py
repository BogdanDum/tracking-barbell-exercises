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

# difference between first and last timestamps
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

# compute average duration
for s in df["set"].unique():
    first = df[df["set"] == s].index[0]
    last = df[df["set"] == s].index[-1]
    duration = last - first
    df.loc[(df["set"] == s), "duration"] = duration.seconds

df_duration = df.groupby(["category"])["duration"].mean()
df_duration.iloc[0] / 5   # duration of 1 medium rep
df_duration.iloc[1] / 10  # duration of 1 heavy rep



# Use the Butterworth lowpass filter
df_lowpass = df.copy()
lowpass = LowPassFilter()
fs = 1000 / 200 # 5 instances per second
cutoff = 1

# apply the filter to a single value
df_lowpass = lowpass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order = 5)

subset = df_lowpass[df_lowpass["set"] == 28]
print(subset["label"][0])

# set up multiple plots
fix, ax = plt.subplots(nrows = 2, sharex=True, figsize = (20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop = True), label = "Raw Data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop = True), label = "butterworth filter")
ax[0].legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), fancybox = True, shadow = True)
ax[1].legend(loc = "upper center", bbox_to_anchor = (0.5, 1.15), fancybox = True, shadow = True)





