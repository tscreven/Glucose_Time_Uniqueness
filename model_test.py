import os
import argparse
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from model_methods import *
                                
def parse_args():
    parser = argparse.ArgumentParser(description='Time in day detection')
    parser.add_argument('data', help='Filepath to testing data.')
    parser.add_argument('load_model', help='Filepath to trained_model.')
    parser.add_argument('-file_results', default=None, type=str, help='CSV file holding results.')
    parser.add_argument('-visualize',help='Visualize training and validation accuracy.',action='store_true')
    parser.add_argument('-seed', default=34, type=int, help='Seed number.')
    return parser.parse_args()
         
if __name__ == '__main__':
    s_time = time.time()

    clear_session()
    args = parse_args()
    tf.random.set_seed(args.seed) 

    X_test = np.load(os.path.join(args.data, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data, 'y_test.npy'))

    model = load_model(args.load_model)
    scores = model.predict(X_test)
    if args.file_results is not None and not os.path.exists(args.file_results):
        write_file_header(args.file_results, training_result=False)
    predictions = metrics(y_test, scores, args.file_results, verbose=True)
    prefix= args.load_model[:-12] #removing "_model.keras" filepath to model.
    if args.visualize:
        get_cf_matrix(y_test, predictions, savepath=f'{prefix}_cfmatrix_testing.png', show=False)
    
    e_time = time.time()
    print(f'Time to test model = {round(e_time-s_time, 2)} seconds.')
