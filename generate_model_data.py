import sys
import json
import numpy as np
import argparse
import os
from statistics import median
import pandas as pd

STRIDE = 0
CHUNK_SIZE = 0

#
# Break inputed data list into chunks using STRIDE and CHUNK_SIZE.
#
def break_chunks(data):
    return [data[i:i+CHUNK_SIZE] for i in range(0, len(data) - CHUNK_SIZE + 1, STRIDE)]

# 
# Return list of data after putting it through min max normalization.
#
def normalize(data):
    min_val = min(data)
    range_data = max(data) - min_val
    return list(map(lambda x: (x - min_val) / range_data, data))

# 
# Return time in number of minutes since midnight.
# 
def convert_timestamp(time:str) ->int:
    hour, minute = time[-5:].split(':')
    return 60 * int(hour) + int(minute)

# 
# Return list of windows containing times. Return list containing which windows
# to include in dataset. Do not include if a window has consecutive datapoints
# more than 30 minutes apart.
# 
def get_time_chunks(timestamps):
    timestamps = [convert_timestamp(time) for time in timestamps]
    time_chunks = break_chunks(timestamps)
    include_chunks = []
    for chunk in time_chunks:
        prev = chunk[0]
        for i, time in enumerate(chunk):
            if time - prev > 30 and time > prev:
                include_chunks.append(False)
                break
            prev = time
            if i+1 == len(chunk):
                include_chunks.append(True)

    return time_chunks, include_chunks

# 
# Smooth glucose data using min max normalization. Return windows of normalized
# data.
# 
def get_gluc_chunks(glucose):
    return break_chunks(normalize(glucose))

# 
# Calculate delta glucose from normalized glucose data by taking difference from
# consecutive glucose readings. 
#
def get_delta_gluc_chunks(cgm_glucose):
    delta_glucose = []
    for i in range(len(cgm_glucose) - 1):
        delta_glucose.append(cgm_glucose[i+1] - cgm_glucose[i])
    delta_glucose.append(-1) # no value to subtract for last cgm reading
    return break_chunks(delta_glucose)

# 
# Calculate delta delta glucose of one window of data. Calculate difference of
# consecutive values in sorted list. 
# 
def get_d_d_gluc_chunks(delta_gluc_chunks):
    delta_delta_gluc = []
    for chunk in delta_gluc_chunks:
        delta_delta_gluc.append([chunk[k+1] - chunk[k] for k in range(len(chunk) - 1)])
        delta_delta_gluc[-1].append(-1)
    return delta_delta_gluc

#
# For each glucose reading in a window, compute difference between glucose
# value and the median value for that window. Measure to find amount of
# variation in glucose in a window. 
#
def get_median_chunks(glucose_chunk):
    median_diff_chunk = []
    for chunk in glucose_chunk:
        median_diff_chunk.append([x - median(chunk) for x in chunk])
    return median_diff_chunk

#
# For each glucose reading in a window, compute if glucose value is within
# bound_low and bound_high. Measure to find extreme glucose values in window.
#
def get_outlier_chunks(glucose, bound_low, bound_high):
    outliers = []
    for g in glucose:
        outliers.append(int(g >= bound_low and g <= bound_high))
    return break_chunks(outliers)

#
# Based on inputed feature flags, return formatted input data and labels for ML
# model.
#
def process_data(use_glucose:bool, use_delta:bool, use_delta_delta:bool, use_median:bool, use_outlier:bool):
    x = []
    y = []
    label1_proportion = 0.99
    files = os.listdir('Data/JSON')
    for file in files:
        with open(f'Data/JSON/{file}') as f:
            data = json.load(f)
    
        bg = data['BG']    
        timestamps = data['Timestamp']

        time_chunks, include_chunks = get_time_chunks(timestamps)
        bg_chunks = get_gluc_chunks(bg)
        d_chunks = get_delta_gluc_chunks(bg)
        d_d_delta_chunks = get_d_d_gluc_chunks(d_chunks)
        diff_chunks = get_median_chunks(bg_chunks)
        bound1, bound2 = list(pd.DataFrame(bg).quantile([0.15, 0.85])[0])
        outlier_chunks = get_outlier_chunks(bg, bound1, bound2)

        assert len(bg_chunks) == len(d_chunks) == len(d_d_delta_chunks) == len(diff_chunks) == len(time_chunks) == len(include_chunks)
        num_chunks = len(bg_chunks)

        for i in range(num_chunks):
            if not include_chunks[i]: continue

            time_chunk = time_chunks[i]
            if time_chunk[0] >= 60 and time_chunk[-1] <= 540:
                y.append(1)
            else:
                y.append(0)

            row = []
            if use_glucose:
                row.append(bg_chunks[i])
            if use_delta:
                row.append(d_chunks[i])
            if use_delta_delta:
                row.append(d_d_delta_chunks[i])
            if use_median:
                row.append(diff_chunks[i])
            if use_outlier:
                row.append(outlier_chunks[i])

            x.append(row)
    
    x = np.array(x)
    y = np.array(y)
                
    num_class_0 = np.sum(y == 0)
    num_class_1 = np.sum(y == 1)

    # Calculate class imbalance.
    num_needed = abs(num_class_0 - num_class_1)
    majority_class = 0 if num_class_0 > num_class_1 else 1

    if num_needed > 0:
        indices_min_class = np.where(y == majority_class)[0] # Randomly sample indices to remove from class 0.
        indices_to_remove = np.random.choice(indices_min_class, num_needed, replace=False)
        mask = np.ones(len(y), dtype=bool)
        mask[indices_to_remove] = False
        X_balanced = x[mask]
        Y_balanced = y[mask]
        x = X_balanced
        y = Y_balanced
        print(f"Balancing complete. Removed {num_needed} from class {majority_class}. Number of samples in each class = {np.sum(Y_balanced == 0)}")
    else:
        print("No balancing needed. Classes are already balanced.")

    x = np.transpose(x, (0, 2 , 1))

    return x, y

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate train and test data')
    parser.add_argument('directory_path',help='Path to directory where numpy files will be stored',default='.')
    parser.add_argument('-split',help='Percentage of data in train test',type=float, default=0.8)
    parser.add_argument('-stride',help='Stride length of window',type=int)
    parser.add_argument('-chunk_size',help='Window length',type=int)
    parser.add_argument('-use_gluc',help='Use glucose as a feature',action='store_true')
    parser.add_argument('-use_d_gluc',help='Use delta glucose as a feature',action='store_true')
    parser.add_argument('-use_d_d_gluc',help='Use delta delta glucose as a feature',action='store_true')
    parser.add_argument('-use_dev',help='Use glucose median differential as a feature',action='store_true')
    parser.add_argument('-use_outlier',help='Use glucose outliers as a feature',action='store_true')
    args = parser.parse_args()

    features = []
    if args.use_gluc: features.append('glucose')
    if args.use_d_gluc: features.append('delta glucose')
    if args.use_d_d_gluc: features.append('delta delta glucose')
    if args.use_dev: features.append('glucose deviation')
    if args.use_outlier: features.append('glucose outliers')

    if len(features) == 0:
        print('Must include at least one feature.')
        sys.exit(1)

    if not os.path.exists(args.directory_path):
        os.makedirs(args.directory_path)

    print(f'Using stride {args.stride} and window size {args.chunk_size}')
    print(f'Including feature(s) {", ".join(features)}')

    STRIDE = args.stride
    CHUNK_SIZE = args.chunk_size
    x, y = process_data(args.use_gluc, args.use_d_gluc, args.use_d_d_gluc, args.use_dev, args.use_outlier)
    
    p = np.random.permutation(len(x))
    x_shuffle = x[p]
    y_shuffle = y[p]
    split = round(args.split * len(x))
    x_train = x_shuffle[:split]
    y_train = y_shuffle[:split]
    x_test = x_shuffle[split:]
    y_test = y_shuffle[split:]
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    np.save(f'{args.directory_path}/X_train.npy', x_train)
    np.save(f'{args.directory_path}/X_test.npy', x_test)
    np.save(f'{args.directory_path}/y_train.npy', y_train)
    np.save(f'{args.directory_path}/y_test.npy', y_test)
