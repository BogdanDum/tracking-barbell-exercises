# Model that can track anc count the number of reps for 5 gym exercises

The model that I have created can detect 5 barbel exercises and count the number of reps for them: 
- **Bench press**
![alt text](https://www.shutterstock.com/image-vector/man-doing-barbell-bench-press-chest-1841766727)

- **Squat**
![alt text](https://www.shutterstock.com/image-vector/man-doing-smith-machine-barbell-squat-2388644813)

- **Overhead Press**
![alt text](https://www.shutterstock.com/image-vector/man-doing-smith-machine-barbell-squat-2388644813)

- **Deadlift**
![alt text](https://www.shutterstock.com/image-vector/man-doing-sumo-barbell-deadlifts-exercise-2034318965)

- **Row**
![alt text](https://www.shutterstock.com/image-vector/man-doing-bentover-barbell-rows-floor-1840374166)

The model can also detect **resting periods**, making it really handy for the user.

The data was collected by participants that wore wristbands while their workouts. The measurements consist of **acceloremeter** (measured in g - G-forcess) and **gyroscope** (measured in deg/s), both on x, y, and z-axis. There was a total of 5 participants who performed **heavy** sets (5 reps) or **medium** sets (10 reps).

Really briefly, the steps I have taken were the following:
- convert the raw data and clean it accordingly
- visualize the data, merge datasets and export the dataset
- plot all exercises, all participants, compare heavy vs medium sets, plot multiple axis and export for both sensors
- detected outliers by appling IQR, Chauvenet’s Criterion and LOF (Chauvenet’s Criterion turned out to be the best one)
- performed feature engineering by dealing with missing values, appling the ButterWorth lowpass filter, contructing principal components (PCA), doing temporal and frequency abstraction, handling overlapping and cluster featuring
- Train model: split into train-test data, create feature subsets, forward feature selecting (decision tree), grid search for best hyperparameters, use the best model and evaluate the results 
- Set up counting the reps: view data for patterns, apply and tweak the lowpass filter, create function for count and a benchmark dataframe and evaluate the final results

For identifying the exercise, the model came up to an accuracy of approximately 99%, which is great! Additionally, the rep counting is off by an average of approximately 1 rep. Further work will go on on this to try and perfect the model's performance. 

