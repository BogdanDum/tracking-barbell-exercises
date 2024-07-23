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