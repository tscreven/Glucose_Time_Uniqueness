import os
import csv
import json

csv_directory = 'Data/ClarityCSV'
json_directory = 'Data/JSON'

# Create JSON files containing the blood glucose, timestamp, and transmitter ID
# of each reading.
def create_json(filepath, json_name):
    data = {
        'BG':[],
        'Timestamp': [],
        "Transmitter ID": []
        }
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == '\ufeff"Index"': continue
            if int(row[0]) > 18:
                bg = row[7]
                if bg == 'Low': 
                    bg = 39
                elif bg == 'High':
                    bg = 401
                else:
                    bg = int(row[7])
                    
                data['Timestamp'].append(row[1][:-3]) # cutting off seconds
                data['BG'].append(bg)
                data['Transmitter ID'].append(row[13])
    
    with open(json_name, mode='w') as f:
        json.dump(data, f)

for i, csv_file in enumerate(os.listdir(csv_directory)):
    create_json(f'{csv_directory}/{csv_file}', f'{json_directory}/Data_{i}.json')