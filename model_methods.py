import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#
# Write header for file containing results of training or testing model.
#
def write_file_header(filepath:str, training_result:bool):
    header = 'Overall Accuracy,Recall,Precision,F1'
    if training_result: header = 'Epoch,' + header
    with open(filepath, mode='w') as f:
        print(header, file=f)

#
# Print training/testing model results into file. Return predictions based on
# model's class probabilities.
#
def metrics(y_true, scores, file_results:str, epoch=None, verbose=True, training_result=False):
    predictions = np.argmax(scores, axis=1)
    overall_accuracy = accuracy_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)

    if file_results is not None:
        with open(file_results, mode='a') as f:
            if training_result: print(epoch, end=',', file=f)
            print(overall_accuracy, end=',', file=f)
            print(recall, end=',', file=f)
            print(precision, end=',', file=f)
            print(f1, file=f)

    if verbose:
        if training_result: print('Epoch:', epoch)
        print('Overall Accuracy:', overall_accuracy)
        print('Recall:', recall)
        print('Precision:', precision)
        print('F1:', f1)
    
    return predictions

#
# Generate confusion matrix from results of testing the model.
#
def get_cf_matrix(y_true, predictions, savepath='', show=False):
    matrix = confusion_matrix(y_true, predictions)
    tot_windows = np.sum(matrix)
    matrix_frac = np.array([[matrix[0][0] / tot_windows, matrix[0][1] / tot_windows],
                            [matrix[1][0] / tot_windows, matrix[1][1] / tot_windows]])
    graph = ConfusionMatrixDisplay(matrix_frac)
    graph.plot(cmap='Blues', values_format=".4f")
    if show: plt.show()
    if savepath != '': plt.savefig(savepath)

#
# Split data into training and testing sets.
#
def split_data(x, y, split:float):
    p = np.random.permutation(len(x))
    x_shuffle = x[p]
    y_shuffle = y[p]
    split = round(split * len(x))
    return x_shuffle[:split], y_shuffle[:split], x_shuffle[split:], y_shuffle[split:]