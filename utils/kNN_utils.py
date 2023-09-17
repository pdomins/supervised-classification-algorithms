import pandas as pd
import numpy as np
from utils.data_split import k_fold_split
from utils.confusion_matrix import calculate_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.kNN import kNN
from matplotlib import pyplot as plt


def plot_k_values(average_precisions, save_file):
    data_array = np.array(average_precisions)

    ks = data_array[:, 0]
    precisions = data_array[:, 1]

    plt.plot(ks, precisions, marker='o', linestyle='-', color='lightblue')
    plt.title('Precision vs. k')
    plt.xlabel('k')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(save_file, bbox_inches='tight', dpi=1200)


def get_best_k(df, to_predict, class_labels, k_fold, is_weighted=False):
    start = 1
    stop = 42
    step = 2

    splits = [k_fold_split(df, k_fold) for _ in range(10)]
    print(f"{'weighted ' if is_weighted else ''}kNN")
    average_precisions = []
    for k in range(start, stop, step):
        precisions = []
        for train_df, test_df in splits:
            res = kNN(train_df, test_df, to_predict, k, is_weighted)
            conf_matrix = calculate_confusion_matrix(class_labels, res['predictions'], res[to_predict])
            per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_matrix)
            metrics_result = metrics(per_label_conf_matrix)
            for key, value in metrics_result.items():
                precisions.append(float(value['Precision']))

        average_precision = np.mean(precisions)
        average_precisions.append([k, average_precision])
    plot_k_values(average_precisions, f"{'weighted_' if is_weighted else ''}kNN_k_values.png")
    return max(average_precisions, key=lambda x: x[1])[0]


def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)
