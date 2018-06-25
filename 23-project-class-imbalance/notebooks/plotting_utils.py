import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
import itertools


def plot_overlapping_histograms(df, col, hue, size=7, aspect=1.5, 
                                title=None, bins=None):
    g = sns.FacetGrid(df, hue=hue, size=size, aspect=aspect)
    g = g.map(plt.hist, col, density=True, alpha=0.35, bins=bins)
    g.add_legend()
    g.fig.suptitle(title)
    
    
def countplot_independent_ylims(df, col, hue, size=5, hue_order=None, title=None):
    g = sns.FacetGrid(df, col=hue, sharey=False, size=size)
    g = g.map(sns.countplot, col, order=hue_order)
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(title)


def plot_1d_corr_heatmap(corr: pd.Series, annot=True, fmt='.2f', 
                         cmap='coolwarm'):
    max_corr = corr.abs().max()
    heatmap_df = pd.DataFrame(corr.sort_values(ascending=False))
    plt.subplots(figsize=(1.5, len(corr)//3.5))

    sns.heatmap(heatmap_df, annot=annot, fmt=fmt, cmap=cmap,
                center=0, vmin=-max_corr, vmax=max_corr)


# adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_test, y_pred, 
                          class_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
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


# adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
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
