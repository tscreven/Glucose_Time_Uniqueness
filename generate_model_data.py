import sys
import json
import numpy as np
import argparse
import os
import pandas as pd

STRIDE = 0
WINDOW_SIZE = 0
MAX_GAP = 0.25

#
# Break inputed data list into chunks using STRIDE and WINDOW_SIZE.
#
def break_chunks(data):
    return [data[i:i+WINDOW_SIZE] for i in range(0, len(data) - WINDOW_SIZE + 1, STRIDE)]

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
# to include in dataset. Do not include window if it has too many consecutive
# times without a glucose vlaue, or has overlapping data in target window and
# outside target window.
# 
def get_time_chunks(timestamps, start, end):
    timestamps = [convert_timestamp(time) for time in timestamps]
    time_chunks = break_chunks(timestamps)
    include_chunks, labels = [], []
    time_gap = min(30, MAX_GAP*WINDOW_SIZE*5)

    for chunk in time_chunks:
        after_start = is_in_window(chunk[0], start, end)
        before_end = is_in_window(chunk[-1], start, end)
        labels.append(int(before_end and after_start))
        if after_start ^ before_end:
            include_chunks.append(False)
        else:
            prev = chunk[0]
            for i, time in enumerate(chunk):
                if time - prev > time_gap and time > prev:
                    include_chunks.append(False)
                    break
                prev = time
                if i+1 == len(chunk):
                    include_chunks.append(True)

    return time_chunks, include_chunks, labels

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
# Calculate delta delta glucose by taking difference of consecutive delta
# glucose values. 
# 
def get_d_d_gluc_chunks(delta_gluc_chunks):
    delta_delta_gluc = []
    for chunk in delta_gluc_chunks:
        delta_delta_gluc.append([chunk[k+1] - chunk[k] for k in range(len(chunk) - 1)])
        delta_delta_gluc[-1].append(-1)
    return delta_delta_gluc

#
# Calculate difference between glucose value and the median value for that
# window. Measure to find amount of variation in glucose in a window. 
#
def get_median_chunks(glucose_chunk):
    median_diff_chunk = []
    for chunk in glucose_chunk:
        median_diff_chunk.append([x - np.median(chunk) for x in chunk])
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
# Return true if time is after start and before end. Handles case where start
# and end wrap around the 24 hour clock (eg. start = 23:00, end = 1:00)
#
def is_in_window(time:int, start:int, end:int) ->bool:
    if end > start:
        return time > start and time < end
    if time > start:
        return time > start and time < end + 24*60
    return time < end

#
# Based on inputed feature flags, return formatted input data and labels.
# Window is labeled as positive if all of the window's data is timestamped
# between start and end, vice versa for negative. Window is not used if it has
# data in both the positive and negative class.
#
def process_data(use_glucose:bool, use_delta:bool, use_delta_delta:bool, use_median:bool, use_outlier:bool, start:int, end:int):
    x = []
    y = []
    files = os.listdir('Data/JSON')
    for file in files:
        with open(f'Data/JSON/{file}') as f:
            data = json.load(f)
    
        bg = data['BG']    
        timestamps = data['Timestamp']

        time_chunks, include_chunks, labels = get_time_chunks(timestamps, start, end)
        bg_chunks = get_gluc_chunks(bg)
        d_chunks = get_delta_gluc_chunks(bg)
        d_d_delta_chunks = get_d_d_gluc_chunks(d_chunks)
        diff_chunks = get_median_chunks(bg_chunks)
        bound1, bound2 = list(pd.DataFrame(bg).quantile([0.15, 0.85])[0])
        outlier_chunks = get_outlier_chunks(bg, bound1, bound2)

        assert len(bg_chunks) == len(d_chunks) == len(d_d_delta_chunks) == len(diff_chunks) == len(time_chunks) == len(include_chunks) == len(labels)
        num_chunks = len(bg_chunks)
        for i in range(num_chunks):
            if include_chunks[i]: 
                y.append(labels[i])
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

    # Calculate class imbalance.              
    num_class_0 = np.sum(y == 0)
    num_class_1 = np.sum(y == 1)
    num_needed = abs(num_class_0 - num_class_1)
    majority_class = 0 if num_class_0 > num_class_1 else 1
    if num_needed > 0:
        indices_min_class = np.where(y == majority_class)[0] # Randomly sample indices to remove from class 0.
        indices_to_remove = np.random.choice(indices_min_class, num_needed, replace=False)
        mask = np.ones(len(y), dtype=bool)
        mask[indices_to_remove] = False
        x_balanced = x[mask]
        y_balanced = y[mask]
        x = x_balanced
        y = y_balanced
        print(f"Balancing complete. Removed {num_needed} from class {majority_class}. Number of samples in each class = {np.sum(y_balanced == 0)}")
    else:
        print("No balancing needed. Classes are already balanced.")

    x = np.transpose(x, (0, 2 , 1))

    return x, y

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate train and test data')
    parser.add_argument('directory_path',help='Path to directory where numpy files will be stored',default='.')
    parser.add_argument('start',help='Starting time to predict (in minutes)',type=int)
    parser.add_argument('end',help='Ending time to predict (in minutes)',type=int)
    parser.add_argument('-split',help='Percentage of data in train set',type=float, default=0.8)
    parser.add_argument('-stride',help='Stride length of window',type=int)
    parser.add_argument('-window_size',help='Window length',type=int)
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

    if args.start < 0 or args.start > 24*60 or args.end < 0 or args.end > 24*60:
        print('start and end must = [0, 1440]')
        sys.exit(1)

    if args.split <= 0 or args.split >= 1:
        print('-split must = (0, 1)')
        sys.exit(1)
    
    if args.window_size <= 0 or args.stride <= 0:
        print('-window_size and -stride must be positive.')
        sys.exit(1)

    if not os.path.exists(args.directory_path):
        os.makedirs(args.directory_path)


    print(f'Using stride {args.stride} and window size {args.window_size}')
    print(f'Including feature(s) {", ".join(features)}')

    STRIDE = args.stride
    WINDOW_SIZE = args.window_size
    x, y = process_data(args.use_gluc, args.use_d_gluc, args.use_d_d_gluc, args.use_dev, args.use_outlier, args.start, args.end)
    
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