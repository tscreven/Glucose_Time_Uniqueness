# Using Blood Glucose Data and Deep Learning to Identify Time of Day
A time classifier model which uses blood glucose data from a continuous glucose
monitor (CGM) to predict whether or not a continuous sequence of CGM blood
glucose readings are within a certain time range of the day (ie. between 2am
and 8am). The model is a `TensorFlow` Sequential Conv1D binary classifier model
which takes in time series inputs.  The model predicts whether or not a series
of continuous glucose readings are inside or outside a time range. The user
determines the start and end time of this range. Each series contains a
sequence of consecutive data derived from a CGM reading. </br> 

The inspiration for this project was to see if ML models can detect differences
in blood glucose behavior during different times of the day.

## Outline
[Overview](#overview)</br>
[Data Pre-Processing](#downloading-and-processing-data)</br>
[Generate Training and Testing Data](#generate_model_datapy)</br>
[Training Model](#model_trainpy)</br>
[Testing Model](#model_testpy)</br>
[Optional Files](#optional-files)</br>

## Overview
This repo contains Python and shell script files which process downloaded CSV
data files from Dexcom Clarity into JSON files. Data does not have to come from
Dexcom Clarity; instructions are provided
[later](#if-cgm-data-is-not-from-dexcom-clarity) on the needed data format for
other data sources. If you downloaded data from Dexcom Clarity, run `setup.sh`
to configure data into appropriate folders and process the data from Dexcom
Clarity into JSON files.  `generate_model_data.py` generates the training and
testing data for the model. It derives input features from the blood glucose
which the user has the option to use or not use. The user dictates the block of
continuous time (on a 24 hour clock) in which is considered the positive class,
while every time that falls out of this block is the negative class. The user
also decides the size of each window of data (number of continuous CGM
readings) and the stride which is taken from the start of the previous window
to the next window. </br> `model_train.py` trains the `TensorFlow` Sequential
Conv1D binary classifier model to predict whether or not a window of data is
inside the target time range. `model_test.py` tests the model trained in
`model_train.py`.

## Downloading and Processing Data
### If CGM Data is from Dexcom Clarity:</br>
As of July, 2025, the export file formats from Dexcom Clarity match what the
files below are expecting. However, if the format changes and it is no longer
compatible, follow the instructions in [If CGM Data is not from Dexcom Clarity](#if-cgm-data-is-not-from-dexcom-clarity).</br> 
* On the Dexcom Clarity website (clarity.dexcom.com), download the data into
  CSV files. 
* Dexcom Clarity only allows for at most 90 days of data to be downloaded. If
  you want to use more than 90 days of data, generate more than one CSV file
  to capture all of the data.
* Move file(s) into the directory which cloned this repository.
* On command-line in the cloned directory, enter these two commands: </br>
    1) `chmod +x setup.sh`
    2) `./setup.sh`
* These steps should only be executed once. Unless more data is downloaded, then these steps should be redone.

`setup.sh` creates several subdirectories, moves the downloaded CSV files into
one of those subdirectories, and then internaly runs `process_data.py` (you do
not have to manually run this Python file). `process_data.py` collects each CGM
reading's: blood glucose value, timestamp, and transmitter ID (optional). For
every CSV file it creates a cooresponding JSON file which holds these values.
</br> **Optional Change:** `process_data.py` handles Dexcom's "LOW" (when glucose < 40) and "HIGH" (when
  glucose > 400) by equating "LOW" to a glucose value of 39 and "HIGH" to a
  glucose value of 401. If you want to change this to a different number, change the values on line
  25 and line 27 in `process_data.py`. However, these new values must be
  numbers (integer or float).

### If CGM Data is not from Dexcom Clarity</br>
Model requires data from a CGM which autonomously and frequently takes
continuous glucose readings. There cannot be too many breaks in the data
timewise.  `generate_model_dataset.py` expects data in this format:

* Data must be inside JSON file(s) inside the following filepath hierarchy in
  the directory: `./Data/JSON/`.
* Within each JSON file, the data must be in order timewise.
    * No glucose readings listed before a reading which happened before it.
* In each JSON file, there needs to be two key-value pairs inside one JSON object. 
    * One key is "BG" and its value is a JSON array literal of numbers containg
      the blood glucose values. 
    * Next key is "Timestamp" and its value is a JSON array literal of
      cooresponding timestamps to the blood glucose readings. 
    * Each timestamp is a string and must follow the following syntax:
      "`year`-`month`-`day`T`time`". For example, a valid timestamp is:
      "2022-01-03T00:04". This means a blood glucose reading occured on Janary
      3rd at 12:04 am (or 00:04 on a 24 hour clock).
        * Note: time in the timestamp must be on the 24 hour clock, and not
          include seconds.
    * There has to be a timestamp for every glucose value.
    * There can be other key-value pairs in the JSON object as long as the
      glucose values are "BG" and its cooresponding timestamps are in
      "Timestamp"

## generate_model_data.py
### Summary
Instructions to run file in [Command-line section](#command-line).</br>
Generates training and testing data for model. Input data is timeseries windows
of feature inputs derived from consecutive CGM readings. `window_size` is
inputed by user as the number of consecutive CGM readings. For each JSON file,
glucose values are read in order and broken up into chunks of data. After a
chunk of `window_size` consecutive CGM readings are collected, if there is a
time gap between consecutive readings in the window ≥ $\frac{window\_size}{4}$
$\times$ $5$, the window is discarded due to the lack of time continuity
between readings. (Note this assumes 5 minutes between readings, can change this
time in file if CGM uses different frequency).  Additionally, if a portion of
the window is in the target time range and the rest of it is outside that
range, the window is discarded. `stride` dictates how many readings ahead of
previous window's first reading to collect next chunk of CGM readings.   

Every window has up to five and at least one timeseries features derived from
blood glucose data:
* `Glucose`: For each CGM reading in window, smooth blood glucose value through
  min-max normalization.
* `Delta Glucose`: For each CGM reading in window, difference between reading's
  min-max normalized glucose reading and the next normalized glucose reading.
  Last CGM reading in window counted as -1. Measure to find rate of change in
  glucose.
* `Delta Delta Glucose`: For each CGM reading in window, difference between
  reading's delta glucose value and the next reading's delta glucose value.
  Last CGM reading in window counted as -1. Measure inspired by taking second
  derivative of a curve where can gain insight by seeing how delta glucose
  changes.
* `Deviation`: For each CGM reading in window, difference between smoothed
  glucose value and the median smoothed glucose value in the window.
* `Outliers`: For each CGM reading in window, determine if unnormalized glucose
  value (not smoothed) is within a lowerbound percentile (`outlier_low`) and
  upperbound percentile (`outlier_high`) of the raw glucose values from the
  entire JSON file. If it is outside of these bounds, it's considered an
  outlier reading. 

### Command-line
`python3 generate_model_data.py <directory_path> <start> <end> <window_size> <stride> <-split> <-outlier_low> <-outlier_high> <-use_gluc> <-use_d_gluc> <-use_d_d_gluc> <-use_dev> <-use_outlier> <-equal_class_size>`
</br>**Note**: Do not include < and > characters in command-line arguments. Used here to help emphasize each argument.

* `<directory_path>`: Filepath to where training and testing files will be stored.
* `<start>`: Starting time to target range, measured in minutes since midnight
  (00:00). For example, the time 4:30 is equivelant to 270 minutes ($6 \times
  40 + 30=270$)
* `<end>`: Ending time to target range, measured in minutes since midnight
  (00:00). Same minutes logic as `start`. Note: the start and ending time can
  wrap around days (eg. start time = 22:00 and end time = 3:00), but cannot
  extend for more than 24 hours.
* `<window_size>`: Integer dicating number of CGM readings in each window. 
* `<stride>`: Integer dicating how many readings ahead of previous window's
  first reading to collect next chunk of CGM readings. If `stride` <
  `window_size`, there will be overlapping data between windows.
* `<-split>`: Optional argument to set percentage of data in training set. Any
  data not used in training set will be put into the testing set. Default value
  = 0.8.
* `<-outlier_low>`: Optional argument to set lowerbound percentile bound for
  outlier feature.
* `<-outlier_high>`: Optional argument to set upperbound percentile bound for
  outlier feature.
* `<-use_gluc>`: Optional argument to use glucose as feature.
* `<-use_d_gluc>`: Optional argument to use delta glucose as feature.
* `<-use_d_d_gluc>`: Optional argument to use delta delta glucose as feature.
* `<-use_dev>`: Optional argument to use deviation as feature.
* `<-use_outliers>`: Optional argument to use outliers as feature.
* `<-equal_class_size>`: Optional argument to enforce equal class balance in
  dataset. If set, windows will be randomly removed from majority class until equal class representation in dataset is reached.

**TODO**
* If you are using data from a CGM whose gap between readings is not 5 minutes, change `CGM_TIME` on line 11 to reflect the correct frequency in minutes. (If frequency is not whole minutes, can make it into a float value of number of minutes).
* If you want to change the amount of allowable gap time between consecutive readings, change `MAX_GAP` on line 10. Note: `MAX_GAP` must $= [0, 1]$ because it is a fraction.

## model_train.py
`python3 model_train.py <data> <-model_name> <-model_dir> <-trace_len> <-dim> <-epochs> <-batch_size> <-lr> <-momentum> <-dropout_prob> <-file_results> <-visualize> <-seed>`
* `<data>`: Filepath to training data.
* `<-model_name>`: Name of trained model. Default value = "generic_model".
* `<-model_dir>`: Filepath to where trained keras model will be saved. Default value = "models".
* `<-trace_len>`: Number of CGM readings in each window.
* `<-dim>`: Number of features (also dimensions) in each window.
* `<-epochs>`: Number of training epochs. 
* `<-batch_size>`: Batch size.
* `<-lr>`: Learning rate.
* `<-momentum>`: SGD optimizer momentum value.
* `<-dropout_prob>`: Dropout probability.
* `-file_results`: Optional argument to give a CSV filename to save training results. Prints following results into file at the epoch with highest validation accuracy: epoch number, accuracy, recall, precision, and F1 score.
* `<-visualize>`: Optional argument to generate and save 2 plots. The first plot graphs the training and validation accuracy over each training epoch. The second plot is a confusion matrix.
* `<-seed>`: Optional argument to set the seed number.

**TODO**
* Consider tuning the model and changing the model's architecture (add/remove
  layers). The architecture and hyperparameters were chosen to maximize
  accuracy on my own Dexcom data. Possible that different people's glucose
  data/trends neccessitates a different model.
*  `TensorFlow` Sequential Conv1D documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D

## model_test.py
`python3 model_test.py <data> <load_model> <-file_results> <-visualize> <-seed>`
* `<data>`: Filepath to testing data.
* `<load_model>`: Filepath to trained model.
* `-file_results`: Optional argument to give a CSV filename to save testing results. Prints following results into file: accuracy, recall, precision, and F1 score.
* `<-visualize>`: Optional argument to generate a confusion matrix.
* `<-seed>`: Optional argument to set the seed number.

## Optional Files
* `run_model_extensive.sh`: Shell script which extensively tests the time
  classifier model's performance on every combinations of input features over a
  range of target times.
    * If running this file for the first time, enter on command-line: "`chmod
      +x run_model_extensive.sh`". Run file on command-line by entering
      "./run_model_extensive.sh".
* `evaluate_results.py`: Python script which averages metrics in CSV file
  created from `model_train.py` and/or `model_test.py`.