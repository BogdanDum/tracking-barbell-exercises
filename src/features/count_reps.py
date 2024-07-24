import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2




# Load the data
df = pd.read_pickle("../../data/interim/processed_data_01.pkl")

# take out the resting period
# doesn't make sense to count reps on it
df = df[df["label"] != "rest"]

# compute sum of squares again
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)




# Split the data wrt exercises
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]



# Visualize data to spot patterns
plot_df = bench_df
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()




# Set up LowPassFilter to eliminate noise
fs = 1000 / 200
lowpass = LowPassFilter()



# Apply LowPassFilter
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot() # heavy set (5 reps)

col = "acc_r"
lowpass.low_pass_filter(dead_set, col = col, sampling_frequency= fs, cutoff_frequency=0.4, order=5)[col + "_lowpass"].plot()




# Create function to count reps
# count mins and maxes to count reps
def count_reps(dataset, cutoff = 0.4, order = 10, col = "acc_r"):
    data = lowpass.low_pass_filter(dataset, col = col, sampling_frequency= fs, cutoff_frequency=cutoff, order=order)

    # indexes of the peaks
    indexes = argrelextrema(data[col + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]
    
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{col}_lowpass"])
    plt.plot(peaks[f"{col}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{col}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()
    
    return len(peaks)

count_reps(dead_set)

