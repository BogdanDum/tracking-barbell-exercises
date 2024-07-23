import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("/Users/bogdanduminica/Desktop/tracking-barbell-exercises/data/interim/data_features_03.pkl")



# Separate training and test set

# drop some columns, not necessary for now
df_train = df.drop(["participant", "category", "set"], axis = 1)

X = df_train.drop("label", axis = 1)
y = df_train["label"]

# now do the split
# use the stratify to have diverse labels as outputs, to not be repetitive
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify=y)

# visualize the split
fig, ax = plt.subplots(figsize = (10, 5))
df_train["label"].value_counts().plot(kind = "bar", ax = ax, color = "lightblue", label = "Total")
y_train.value_counts().plot(kind = "bar", ax = ax, color = "royalblue", label = "Test")
plt.legend()
plt.show()




# Split current features into subsets
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if ("_freq_" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

print("Basic features:", len(basic_features))
print("Square features:", len(square_features))
print("PCA features:", len(pca_features))
print("Time features:", len(time_features))
print("Frequency features:", len(freq_features))
print("Cluster features:", len(cluster_features))

# use set to avoid duplicate columns
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))
# I will use these to make different data selections




# Do forward feature selection making use of a simple decicion tree
learner = ClassificationAlgorithms()
max_features = 10
# loop over all columns and train a decision tree
selected_features, ordered_features, ordered_cores = learner.forward_selection(max_features, X_train, y_train)

selected_features = ['pca_1',
                    'duration',
                    'acc_z_freq_0.0_Hz_ws_14',
                    'acc_y_temp_mean_ws_5',
                    'gyr_x_freq_1.071_Hz_ws_14',
                    'gyr_y_freq_2.143_Hz_ws_14',
                    'acc_z_freq_1.071_Hz_ws_14',
                    'acc_r_freq_0.714_Hz_ws_14',
                    'gyr_y_freq_1.429_Hz_ws_14',
                    'gyr_y_freq_0.714_Hz_ws_14']

# visualize the results
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_cores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

