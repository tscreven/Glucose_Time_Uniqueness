import csv
import argparse

#
# For each combination of features results in file, write the averaged metrics
# to newfile.
#
def average_results(num_runs:int, file:str, newfile:str):
    file1 = open(file, 'r')
    file2 = open(newfile, 'w')
    reader = csv.reader(file1)
    print('Feature,Overall Accuracy,Recall,Precision,F1', file=file2, end='')

    acc, recall, precision, f1_score = 0, 0, 0, 0
    for r, line in enumerate(reader):
        if r == 0: continue
        acc += float(line[1])
        recall += float(line[2])
        precision += float(line[3])
        f1_score += float(line[4])
        if r % num_runs == 0:
            print(f'\n{line[0]},{acc/num_runs},{recall/num_runs},{precision/num_runs},{f1_score/num_runs}', file=file2, end='')
            acc, recall, precision, f1_score = 0, 0, 0, 0

    file1.close()
    file2.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Average results of multiple model evaluations.')
    parser.add_argument('num_evaluations',help='Number of times model was evaluated for each feature combination', type=int)
    parser.add_argument('filename',help='Filepath to CSV file containing unaveraged model results.')
    parser.add_argument('new_filename',help='Filepath to CSV file containing unaveraged model results.')
    args = parser.parse_args()

    if args.filename[-4:] != '.csv' or args.new_filename[-4:] != '.csv':
        raise Exception(f'Both inputted files need to be .csv files.')

    average_results(args.num_evaluations, args.filename, args.new_filename)