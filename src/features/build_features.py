import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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
cutoff = 1.3

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


# apply filter to all columns
for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    
    # overwrite the original column
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]



# PCA (Principal component analysis)
df_pca = df_lowpass.copy()
pca = PrincipalComponentAnalysis()

pc_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)

# apply elbow technique

# plot computed values on a graph
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Component number")
plt.ylabel("Explained Variance")
plt.show()
# quite obvious from the graph there should be 3 principal components

df_pca = pca.apply_pca(df_pca, predictor_columns, 3)
# basically summarized the 6 variables into 3 principal components

subset = df_pca[df_pca["set"] == 28]
subset[["pca_1", "pca_2", "pca_3"]].plot()

df_pca



# Now use the sum of squares attribute
# magnitude r is sqrt(x^2 + y^2 + z^2)

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2 
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 28]
subset[["acc_r", "gyr_r"]].plot(subplots = True)

df_squared




# Temporal abstraction (compute rolling average)
df_temporal = df_squared.copy()
numabs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# establish window size
ws = int(1000 / 200)

# compute mean and std
for col in predictor_columns:
    df_temporal = numabs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = numabs.abstract_numerical(df_temporal, [col], ws, "std")
    
df_temporal

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = numabs.abstract_numerical(subset, [col], ws, "mean")
        subset = numabs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()




# Deal with the frequency features
# make use of Fourier Transformation

df_freq = df_temporal.copy().reset_index()
freqabs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)

# firstly on a single column
df_freq = freqabs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# see results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []
# made this more efficient by just adding the predictor_columns variable
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop = True).copy()
    subset = freqabs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
    
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop = True)




# Deal with overlapping windows

df_freq = df_freq.dropna()

# reduce overfitting of our model
df_freq = df_freq.iloc[::2]




# Clustering 

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters = k, n_init = 20, random_state = 0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# 4 or 5?
# 5 seems to be the most optimal number of clusters

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# visualize to see if the clusters make sense
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection = "3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.legend()
plt.show()

# now split by label (exercise) instead of cluster and compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection = "3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
plt.legend()
plt.show()



# Export dataset
df_cluster.to_pickle("../../data/interim/data_features_03.pkl")