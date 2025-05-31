#!/bin/bash

# On commandline, enter: chmod +x setup.sh. Then enter: ./setup.sh. Only need
# to run this file once. 
mkdir Data
mkdir Data/ClarityCSV
mkdir Data/JSON
mv *.csv Data/ClarityCSV
python3 process_data.py