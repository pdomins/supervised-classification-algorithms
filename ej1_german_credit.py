import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json

from utils.confusion_matrix import calculate_relative_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.data_split import k_fold_split
from utils.decision_tree import id3, decision_trees_over_possible_depths
from utils.plotter import plot_confusion_matrix, plot_multiple_dicts
from utils.random_forest import RandomForest
from utils.precision_curve import precisions_over_possible_depth



def discretize_variables(df: pd.DataFrame, var: str, bins_amount: int):
    min_val = df[var].min()
    max_val = df[var].max()
    bin_width = (max_val - min_val) / bins_amount
    bin_edges = [min_val + i * bin_width for i in range(bins_amount + 1)]
    bin_labels = [i for i in range(bins_amount)]
    df[var] = pd.cut(df[var], bins=bin_edges, labels=bin_labels, include_lowest=True)


def generate_confusion_matrix(df: pd.DataFrame, predictions_label: str, to_predict_label: str, output_filename: str, possible_out_values : list[str] = None, plot_matrix : bool = True):
    if possible_out_values is not None:
        class_labels = np.array(possible_out_values)
    else:
        class_labels = np.array(df[to_predict_label].unique())
    conf_mat = calculate_relative_confusion_matrix(class_labels, df[predictions_label].to_dict(),
                                                   df[to_predict_label].to_dict())
    per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_mat)
    if (plot_matrix):
        plot_confusion_matrix(conf_mat, "Matriz de confusión", output_filename, ".2f")
    metrics_result = metrics(per_label_conf_matrix)
    for key, value in metrics_result.items():
        precision = value['Precision']
        print(f'{"Devuelve" if (key == 1) else "No devuelve"}: Precision = {precision}')
        accuracy = value['Accuracy']
    print('\n')
    return accuracy


def split_df(df : pd.DataFrame, use_seed : bool = False):
    random_state = None
    if (use_seed):
        random_state = np.random.default_rng(seed=42)
    discretize_variables(df, 'Credit Amount', 5)
    discretize_variables(df, 'Duration of Credit (month)', 4)
    discretize_variables(df, 'Age (years)', 3)
    return k_fold_split(df, k=3, random_state=random_state)


def run_ej1_tree(df : pd.DataFrame, train_df : pd.DataFrame, test_df : pd.DataFrame):
    dec_tree = id3(train_df, "Creditability")
    predicted_column_dt = test_df.apply(dec_tree.predict, axis=1)

    test_df.insert(loc=1, column="Creditability (predicted by DT)", value=predicted_column_dt)
    # test_df.to_csv("ej1_german_credit_prediction.csv", sep=";")

    correct_predictions_dt = test_df[test_df['Creditability (predicted by DT)'] == test_df['Creditability']].shape[0]
    incorrect_prediction_dt = test_df.shape[0] - correct_predictions_dt

    print({
        "test decision tree": {
            "correct": correct_predictions_dt,
            "incorrect": incorrect_prediction_dt
        }
    })

    possible_out_labels = df['Creditability'].unique()

    attrs_vals = dict()
    for column in df.columns:
        attrs_vals[column] = list(df[column].unique())

    generate_confusion_matrix(test_df, predictions_label='Creditability (predicted by DT)', to_predict_label='Creditability', output_filename="./graphics/ej1_conf_mat_dt.png", possible_out_values=possible_out_labels)
    precision_by_label = precisions_over_possible_depth(train_df, test_df, possible_out_labels, "Creditability", attrs_vals)
    precision_by_label_train = precisions_over_possible_depth(train_df, train_df, possible_out_labels, "Creditability", attrs_vals)
    plot_multiple_dicts(precision_by_label, "Precisión vs. cantidad de nodos (test)", "Cantidad de nodos", "Precisión", ylim=(-0.05, 1.05), label_mapping={0: "No otorga", 1 : "Otorga"}, save_file="precision_over_node_test.png")
    plot_multiple_dicts(precision_by_label_train, "Precisión vs. cantidad de nodos (train)", "Cantidad de nodos", "Precisión", ylim=(-0.05, 1.05), label_mapping={0: "No otorga", 1 : "Otorga"}, save_file="precision_over_node_train.png")



def run_ej1_forest(df : pd.DataFrame, train_df : pd.DataFrame, test_df : pd.DataFrame, bag_count : int, sample_size : int):
    attrs_vals = dict()
    for column in df.columns:
        attrs_vals[column] = list(df[column].unique())

    random_forest = RandomForest(train_df, bag_count=bag_count, sample_size=sample_size, attrs_vals=attrs_vals)
    
    predicted_column_rf = test_df.apply(random_forest.predict, axis=1)

    test_df.insert(loc=1, column="Creditability (predicted by RF)", value=predicted_column_rf)

    # test_df.to_csv("ej1_german_credit_prediction.csv", sep=";")

    correct_predictions_rf = test_df[test_df['Creditability (predicted by RF)'] == test_df['Creditability']].shape[0]
    incorrect_prediction_rf = test_df.shape[0] - correct_predictions_rf
    
    print({
        "test random forest": {
            "correct": correct_predictions_rf,
            "incorrect": incorrect_prediction_rf
        }
    })

    generate_confusion_matrix(test_df, predictions_label='Creditability (predicted by RF)', to_predict_label='Creditability', output_filename="./graphics/ej1_conf_mat_rf.png", possible_out_values=list(df['Creditability'].unique()))



def benchmark_forest(df : pd.DataFrame, train_df : pd.DataFrame, test_df : pd.DataFrame):

    attrs_vals = dict()
    for column in df.columns:
        attrs_vals[column] = list(df[column].unique())

    sample_sizes = []
    bag_counts = []
    accuracy = []

    for i in range(10):
        curr_bag_count = (i+1)*10
        for j in range(15):
            curr_sample_size = (j+1)*100
            sample_sizes.append(curr_sample_size)
            bag_counts.append(curr_bag_count)
            curr_forest = RandomForest(train_df, bag_count=curr_bag_count, sample_size=curr_sample_size, attrs_vals=attrs_vals)
            predicted_column = test_df.apply(curr_forest.predict, axis=1)
            curr_column_name = f"Creditability (predicted by RF{i}b{j}s)"
            test_df.insert(loc=1, column=curr_column_name, value=predicted_column)
            curr_accuracy = generate_confusion_matrix(test_df, predictions_label=curr_column_name, to_predict_label='Creditability', output_filename="", plot_matrix=False)
            accuracy.append(curr_accuracy)

    plot_df = pd.DataFrame({'x': sample_sizes, 'y': accuracy, 'z': bag_counts})
    groups = plot_df.groupby('z')
    plt.figure()
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='-', markersize=12, label=name)
    plt.legend(title="Bag Count")
    plt.title('Sample size vs. Accuracy by Bag Count')
    plt.xlabel('Sample size')
    plt.ylabel('Accuracy')
    plt.ylim((-0.05, 1.05))
    plt.grid(True)
    plt.savefig(f"./graphics/rf_benchmark.png", bbox_inches='tight', dpi=1200)

    plot_df = pd.DataFrame({'x': sample_sizes, 'y': accuracy, 'z': bag_counts})
    groups = plot_df.groupby('z')
    plt.figure()
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='-', markersize=12, label=name)
    plt.legend(title="Bag Count")
    plt.title('Sample size vs. Accuracy by Bag Count')
    plt.xlabel('Sample size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f"./graphics/rf_benchmark (1).png", bbox_inches='tight', dpi=1200)

    plt.figure()
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
    plt.legend(title="Bag Count")
    plt.title('Sample size vs. Accuracy by Bag Count')
    plt.xlabel('Sample size')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f"./graphics/rf_benchmark (2).png", bbox_inches='tight', dpi=1200)

    plt.figure()
    for name, group in groups:
        plt.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)
    plt.legend(title="Bag Count")
    plt.title('Sample size vs. Accuracy by Bag Count')
    plt.xlabel('Sample size')
    plt.ylabel('Accuracy')
    plt.ylim((-0.05, 1.05))
    plt.grid(True)
    plt.savefig(f"./graphics/rf_benchmark (3).png", bbox_inches='tight', dpi=1200)

    max_value = max(accuracy)
    best_idx = accuracy.index(max_value)
    best_bag_count = bag_counts[best_idx]
    best_sample_size = sample_sizes[best_idx]

    return best_bag_count, best_sample_size


def run_ej1():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    train_df, test_df = split_df(df, use_seed=True)
    best_bag_count, best_sample_size = benchmark_forest(df, train_df, test_df)
    print("DATASETS -------------------------------------------------------------------------------")
    print(train_df)
    print(test_df)
    print("DECISION TREE --------------------------------------------------------------------------")
    run_ej1_tree(df, train_df, test_df)
    print("RANDOM FOREST --------------------------------------------------------------------------")
    run_ej1_forest(df, train_df, test_df, best_bag_count, best_sample_size)