import csv
import argparse

#
# For each combination of features results in file, write the averaged overall
# accuracy, negative class accuracy, and positive class accuracy to newfile.
#
def average_results(num_runs:int, file:str, newfile:str):
    f1 = open(file, 'r')
    f2 = open(newfile, 'w')
    reader = csv.reader(f1)
    print('Features, Average Overall Accuracy, Average Class 0 Accuracy, Average Class 1 Accuracy', file=f2, end='')

    acc, acc0, acc1 = 0, 0, 0
    for r, line in enumerate(reader):
        if r == 0: continue
        acc += float(line[2])
        acc0 += float(line[3])
        acc1 += float(line[4])
        if r % num_runs == 0:
            print(f'\n{line[0]},{acc/num_runs},{acc0/num_runs},{acc1/num_runs}', file=f2, end='')
            acc, acc0, acc1 = 0, 0, 0

    f1.close()
    f2.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Average results of multiple model evaluations.')
    parser.add_argument('num_evaluations',help='Number of times model was evaluated for each feature combination', type=int)
    parser.add_argument('filename',help='Filepath to CSV file containing unaveraged model results.')
    parser.add_argument('new_filename',help='Filepath to CSV file containing unaveraged model results.')
    args = parser.parse_args()

    if args.filename[-4:] != '.csv' or args.new_filename[-4:] != '.csv':
        raise Exception(f'Both inputted files need to be .csv files.')

    average_results(args.num_evaluations, args.filename, args.new_filename)