import pandas as pd
import numpy as np
from pprint import pprint

from utils.confusion_matrix import calculate_relative_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.data_split import k_fold_split
from utils.decision_tree import id3
from utils.plotter import plot_confusion_matrix
from utils.random_forest import RandomForest



def discretize_variables(df: pd.DataFrame, var: str, bins_amount: int):
    min_val = df[var].min()
    max_val = df[var].max()
    bin_width = (max_val - min_val) / bins_amount
    bin_edges = [min_val + i * bin_width for i in range(bins_amount + 1)]
    bin_labels = [i for i in range(bins_amount)]
    df[var] = pd.cut(df[var], bins=bin_edges, labels=bin_labels, include_lowest=True)


def generate_confusion_matrix(df: pd.DataFrame, predictions_label: str, to_predict_label: str, output_filename: str, possible_out_values : list[str] = None):
    if possible_out_values is not None:
        class_labels = np.array(possible_out_values)
    else:
        class_labels = np.array(df[to_predict_label].unique())
    conf_mat = calculate_relative_confusion_matrix(class_labels, df[predictions_label].to_dict(),
                                                   df[to_predict_label].to_dict())
    per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_mat)
    plot_confusion_matrix(conf_mat, "Matriz de confusión", output_filename, ".2f")
    metrics_result = metrics(per_label_conf_matrix)
    for key, value in metrics_result.items():
        precision = value['Precision']
        print(f'Key {key}: Precision = {precision}')
        # print(f'{"Devuelve" if (key == 1) else "No devuelve"}: Precision = {precision}')


def split_df(df : pd.DataFrame):
    discretize_variables(df, 'Credit Amount', 5)
    discretize_variables(df, 'Duration of Credit (month)', 4)
    discretize_variables(df, 'Age (years)', 3)
    return k_fold_split(df, k=3)


def run_ej1_tree():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    train_df, test_df = split_df(df)

    dec_tree = id3(train_df, "Creditability")
    predicted_column_dt = test_df.apply(dec_tree.predict, axis=1)

    test_df.insert(loc=1, column="Creditability (predicted by DT)", value=predicted_column_dt)
    # test_df.to_csv("ej1_german_credit_prediction.csv", sep=";")

    correct_predictions_dt = test_df[test_df['Creditability (predicted by DT)'] == test_df['Creditability']].shape[0]
    incorrect_prediction_dt = test_df.shape[0] - correct_predictions_dt

    print(train_df)
    print(test_df)
    print({
        "test decision tree": {
            "correct": correct_predictions_dt,
            "incorrect": incorrect_prediction_dt
        }
    })

    generate_confusion_matrix(test_df, predictions_label='Creditability (predicted by DT)', to_predict_label='Creditability', output_filename="./graphics/ej1_conf_mat_dt.png", possible_out_values=list(df['Creditability'].unique()))



def run_ej1_forest():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    train_df, test_df = split_df(df)

    attrs_vals = dict()
    for column in df.columns:
        attrs_vals[column] = list(df[column].unique())


    random_forest = RandomForest(train_df, bag_count=1, sample_size=10, attrs_vals=attrs_vals)
    # random_forest.print()
    # print("\n-------------------------------------------------------------------------\n")
    
    predicted_column_rf = test_df.apply(random_forest.predict, axis=1)

    test_df.insert(loc=1, column="Creditability (predicted by RF)", value=predicted_column_rf)

    # test_df.to_csv("ej1_german_credit_prediction.csv", sep=";")

    correct_predictions_rf = test_df[test_df['Creditability (predicted by RF)'] == test_df['Creditability']].shape[0]
    incorrect_prediction_rf = test_df.shape[0] - correct_predictions_rf
    
    print(train_df)
    print(test_df)
    print({
        "test random forest": {
            "correct": correct_predictions_rf,
            "incorrect": incorrect_prediction_rf
        }
    })

    generate_confusion_matrix(test_df, predictions_label='Creditability (predicted by RF)', to_predict_label='Creditability', output_filename="./graphics/ej1_conf_mat_rf.png", possible_out_values=list(df['Creditability'].unique()))




def benchmark_forest():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    train_df, test_df = split_df(df)

    attrs_vals = dict()
    for column in df.columns:
        attrs_vals[column] = list(df[column].unique())

    forests : dict[RandomForest] = {}
    predictions = {}

    for i in range(1, 2):
        curr_forest : RandomForest = RandomForest(train_df, bag_count=i, sample_size=10, attrs_vals=attrs_vals)
        predictions[i] = test_df.apply(curr_forest.predict, axis=1)
        forests[i] = curr_forest
        




def run_ej1():
    # return benchmark_forest()
    return run_ej1_forest()