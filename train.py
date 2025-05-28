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
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.backend import clear_session, sum, pow, log
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

def parse_args():
    parser = argparse.ArgumentParser(description='Time in day detection')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--experiment', default='generic_model', type=str, dest='experiment',
                        help='''name of the experiment''')
    parser.add_argument('--num_classes', default=2, type=int, metavar='C', help='number of classes')
    parser.add_argument('--seed', default=42, type=int, metavar='S', help='random seed')
    parser.add_argument('--epochs', default=1000, type=int, metavar='E',
                        help='number of epochs to run training for')
    parser.add_argument('-b', '--batch_size', default=500, type=int, metavar='B', help='batch size')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum value for SGD')
    parser.add_argument('--dropout_prob', default=0, type=float, dest='dropout_prob',
                        help='dropout probability')
    parser.add_argument('--ts_dim', default=4, type=int, dest='ts_dim',
                        help='num dims in each time step')
    parser.add_argument('--trace_length', default=288, type=int, dest='trace_len',
                        help='length of each sequence')
    parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, dest='ckpt_dir',
                        help='path to save checkpoints')
    parser.add_argument('--file_results', default='', type=str, dest='file_results')
    return parser.parse_args()

def conv_model(ts_dim, trace_len, num_classes, dropout_prob):
    # model = Sequential([Conv1D(filters=32, kernel_size=3, activation='relu',
    #                            input_shape=(trace_len, ts_dim), padding='same'),
    #                     Flatten(),
    #                     Dense(32, activation='relu'),
    #                     Dense(num_classes, activation='softmax'),
    #                     Dropout(dropout_prob)])
    model = Sequential([Conv1D(filters=4, kernel_size=3, activation='relu',
                               input_shape=(trace_len, ts_dim), padding='same'),
                        Flatten(),
                        Dense(num_classes, activation='sigmoid')])
    return model

def dispatch():
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

    model = conv_model(args.ts_dim, args.trace_len, args.num_classes, args.dropout_prob)
        
    model.compile(optimizer=SGD(learning_rate=args.lr, momentum=args.momentum),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    #checkpoint_name = args.experiment + '_model.h5'
    checkpoint_name = args.experiment + '_model.keras'
    
    checkpoint = ModelCheckpoint(os.path.join(args.ckpt_dir,
                                              checkpoint_name),
                                 monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    
    history = model.fit(X_train, y_train, epochs=args.epochs,
                        batch_size=args.batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint],
                        class_weight=class_weights)
    
     # Identify epoch with max validation accuracy
    max_val_acc_epoch = np.argmax(history.history['val_accuracy'])

    # Predict at the epoch with the max validation accuracy
    best_model = load_model(os.path.join(args.ckpt_dir, checkpoint_name))
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate class-wise accuracy 
    overall_accuracy = accuracy_score(y_test, y_pred_classes)
    class_0_accuracy = accuracy_score(y_test[y_test == 0], y_pred_classes[y_test == 0])
    class_1_accuracy = accuracy_score(y_test[y_test == 1], y_pred_classes[y_test == 1])

    print(f"Epoch: {max_val_acc_epoch}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Class 0 Accuracy: {class_0_accuracy:.4f}")
    print(f"Class 1 Accuracy: {class_1_accuracy:.4f}")
    if args.file_results != '':
        with open(args.file_results, mode='a') as f:
            print(max_val_acc_epoch, end=',', file=f)
            print(f'{overall_accuracy:.4f}', end=',', file=f)
            print(f'{class_0_accuracy:.4f}', end=',', file=f)
            print(f'{class_1_accuracy:.4f}', file=f)


    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(checkpoint_name + '.png')


if __name__ == '__main__':
    dispatch()
