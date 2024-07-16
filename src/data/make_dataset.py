import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion/"
f = files[0] # first file in folder

# extract parameters from the file names
participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
weight = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["weight"] = weight

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

# used to increment set no to create unique identifier
acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    weight = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(f)
    
    df["participant"] = participant
    df["label"] = label
    df["weight"] = weight
    
    # split files based on acc & gyr
    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

acc_df[acc_df["set"] == 1]

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

# convert object to datetime variable
acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit = "ms")

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")

# delete the rest of the columns related to timestamps
# only keep the current index
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)

def read_data(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    # used to increment set no to create unique identifier
    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        weight = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["weight"] = weight
        
        # split files based on acc & gyr
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
            
        
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit = "ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit = "ms")

    # delete the rest of the columns related to timestamps
    # only keep the current index
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

merged_data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis = 1)

# update the names of the columns
merged_data.columns = ["acc_x", "acc_y", "acc_z",
                       "gyr_x", "gyr_y", "gyr_z",
                       "label", "category", "participant", "set"]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# dictionary, mean for numerical data and last for categorical 
columns = {
    'acc_x' : "mean", 'acc_y' : "mean", 'acc_z' : "mean", 'gyr_x' : "mean", 'gyr_y' : "mean", 'gyr_z' : "mean", 'label' : "last",
       'category' : "last", 'participant' : "last", 'set' : "last"
}

merged_data[:1000].resample(rule = "200ms").apply(columns)

# split by day
days = [g for n, g in merged_data.groupby(pd.Grouper(freq="D"))]

resampled_data = pd.concat([df.resample(rule = "200ms").apply(columns).dropna() for df in days])

resampled_data.info()

# convert float to int for set
resampled_data["set"] =resampled_data["set"].astype("int")



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

resampled_data.to_pickle("../../data/interim/processed_data_01.pkl")