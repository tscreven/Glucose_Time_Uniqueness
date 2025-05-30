#!/bin/bash

# Bash script which extensively tests the time classifier model's performance
# on every combinations of input features for a range of target times.

input_features=("-use_gluc" "-use_d_gluc" "-use_d_d_gluc" "-use_dev" "-use_outlier")
window_length=12 # Number of glucose readings placed in each window of data.
num_runs=5 # Number of times each combination of input features is generated, and trained/tested by the model.
time_increment=120 # Number of minutes to increment from previous target time's end.
target_length=480 #Total number of minutes from start time to end time labeled as positive class.
num_features=${#input_features[@]}
total=$((1 << num_features)) 

for ((s=0; s<=1440; s+=$time_increment)); do
    e=$((${s}+${target_length}))
    filename1=model_runs_${s}_${e}_1h.csv
    filename2=${s}_${e}_1h.csv

    echo Features,Epoch,Overall Accuracy,Recall,Precision,F1 > $filename1

    for ((i=1; i<total; i++)); do
        subset=()
        for ((j=0; j<num_features; j++)); do
            if (( (i >> j) & 1 )); then
                subset+=("${input_features[j]}")
            fi
        done
        
        # NOTE: If you want to include -model_name option to train.py, make
        # sure to also change filepath argument to test.py
        arg="${subset[*]}"
        num_dim=${#subset[@]}
        for ((a=0; a<$num_runs; a++)); do
            printf "%s," "$arg" >> $filename1
            python3 generate_model_data.py . $s $e -stride $window_length -window_size $window_length $arg
            python3 model_train.py . -trace_len $window_length -epochs 2500 -dim $num_dim
            python3 model_test.py . checkpoints/generic_model_model.keras -file_results testing_$filename1
        done
    done

    python3 evaluate_results.py $num_runs $filename1 $filename2
    mv *.csv Results
done
