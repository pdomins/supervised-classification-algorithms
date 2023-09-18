import pandas as pd
import numpy as np

from utils.confusion_matrix import calculate_relative_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.data_split import k_fold_split
from utils.decision_tree import id3
from utils.plotter import plot_confusion_matrix


def discretize_variables(df: pd.DataFrame, var: str, bins_amount: int):
    min_val = df[var].min()
    max_val = df[var].max()
    bin_width = (max_val - min_val) / bins_amount
    bin_edges = [min_val + i * bin_width for i in range(bins_amount + 1)]
    bin_labels = [i for i in range(bins_amount)]
    df[var] = pd.cut(df[var], bins=bin_edges, labels=bin_labels, include_lowest=True)


def generate_confusion_matrix(df: pd.DataFrame, predictions_label: str, to_predict_label: str):
    class_labels = np.array(df[to_predict_label].unique())
    conf_mat = calculate_relative_confusion_matrix(class_labels, df[predictions_label],
                                                   df[to_predict_label])
    per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_mat)
    plot_confusion_matrix(conf_mat, "Matriz de confusión", "./graphics/ej1_conf_mat.png", ".2f")
    metrics_result = metrics(per_label_conf_matrix)
    for key, value in metrics_result.items():
        precision = value['Precision']
        print(f'Key {key}: Precision = {precision}')


def run_ej1():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    discretize_variables(df, 'Credit Amount', 5)
    discretize_variables(df, 'Duration of Credit (month)', 4)
    discretize_variables(df, 'Age (years)', 3)

    train_df, test_df = k_fold_split(df, k=3)
    dec_tree = id3(train_df, "Creditability")
    predicted_column = test_df.apply(dec_tree.predict, axis=1)

    test_df.insert(loc=1, column="Creditability (predicted)", value=predicted_column)
    test_df.to_csv("ej1_german_credit_prediction.csv", sep=";")

    correct_predictions = test_df[test_df['Creditability (predicted)'] == test_df['Creditability']].shape[0]
    incorrect_prediction = test_df.shape[0] - correct_predictions
    print(train_df)
    print(test_df)
    print({
        "test": {
            "correct": correct_predictions,
            "incorrect": incorrect_prediction
        }
    })
    generate_confusion_matrix(test_df, predictions_label='Creditability (predicted)', to_predict_label='Creditability')
