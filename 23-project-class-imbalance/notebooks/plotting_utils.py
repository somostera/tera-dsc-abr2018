import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
import itertools


# adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_test, y_pred, 
                          class_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def precision_recall(y_test, y_score) -> pd.DataFrame:
    """Imprime a curva precision-recall e retorna os dados.
    
    Parâmetros:
        - y_test: os valores verdadeiros.
        - y_score: os scores calculados pelo modelo.
    
    Retorno:
        - um DataFrame contendo precisão, recall e f1-score para cada 
        threshold calculado.
    """
    precision, recall, thresholds = _precision_recall_plot(y_test, y_score)
    return _precision_recall_results(precision, recall, thresholds)


def _precision_recall_plot(y_test, y_score):
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    
    return precision, recall, thresholds


def _precision_recall_results(precision, recall, thresholds):
    results = pd.DataFrame(data=[precision, recall, thresholds],
                           index=['precision', 'recall', 'threshold']).T

    results['f1_score'] = (2 * (results['precision'] * results['recall']) 
                             / (results['precision'] + results['recall']))
    
    return results
