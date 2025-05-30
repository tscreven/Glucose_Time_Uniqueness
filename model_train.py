import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Flatten, concatenate, BatchNormalization,
                                     Dropout, GlobalAveragePooling2D, LSTM, Permute)
from multiprocessing import Pool, cpu_count
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from model_methods import *

def parse_args():
    parser = argparse.ArgumentParser(description='Time in day detection')
    parser.add_argument('data', help='Path to training data.')
    parser.add_argument('-model_name', default='generic_model', help='Name of model.')
    parser.add_argument('-model_dir', default='models', type=str, help='Filepath to save model checkpoints.')
    parser.add_argument('-trace_length', default=12, type=int, help='Number of glucose readings in each window.')
    parser.add_argument('-dim', default=5, type=int, help='Number of feature dimensions in each data window.')
    parser.add_argument('-num_classes', default=2, type=int, help='Number of classes.')
    parser.add_argument('-epochs', default=1000, type=int, help='Number of training epochs.')
    parser.add_argument('-batch_size', default=500, type=int, help='Batch size.')
    parser.add_argument('-lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('-momentum', default=0.9, type=float, help='SGD\'s momentum value.')
    parser.add_argument('-dropout_prob', default=0, type=float, help='Dropout probability.')
    parser.add_argument('-file_results', default=None, type=str, help='CSV file holding results.')
    parser.add_argument('-visualize',help='Visualize training and validation accuracy.',action='store_true')
    parser.add_argument('-seed', default=34, type=int, help='Seed number.')
    return parser.parse_args()
    
def main():
    clear_session()
    args = parse_args()
    tf.random.set_seed(args.seed) 
    
    X_train = np.load(os.path.join(args.data, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data, 'y_train.npy'))

    X_test = np.load(os.path.join(args.data, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data, 'y_test.npy'))

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    model = Sequential([Conv1D(filters=4, kernel_size=3, activation='relu',
                               input_shape=(args.trace_length, args.dim), padding='same'),
                        Flatten(),
                        Dense(args.num_classes, activation='sigmoid')])
        
    model.compile(optimizer=SGD(learning_rate=args.lr, momentum=args.momentum),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    model_name = args.model_name + '_model.keras'
    
    checkpoint = ModelCheckpoint(os.path.join(args.model_dir,
                                              model_name),
                                 monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    
    history = model.fit(X_train, y_train, epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint],
                        class_weight=class_weights)
    
     # Get epoch with max validation accuracy
    max_val_acc_epoch = np.argmax(history.history['val_accuracy'])

    # Predict at the epoch with the max validation accuracy
    best_model = load_model(os.path.join(args.model_dir, model_name))
    scores = best_model.predict(X_test)
    
    if args.file_results is not None and not os.path.exists(args.file_results):
        write_file_header(args.file_results, training_result=True)

    predictions = metrics(y_test, scores, args.file_results, epoch=max_val_acc_epoch, verbose=True, training_result=True)

    if args.visualize:
        name_prefix = f'{args.model_dir}/{args.model_name}'
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(f'{name_prefix}_trainingacc.png')
        #get_cf_matrix(y_test, predictions, savepath=f'{name_prefix}_cfmatrix_training.png')


if __name__ == '__main__':
    main()