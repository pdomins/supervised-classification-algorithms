import pandas as pd
import numpy as np
from utils.data_split import k_fold_split
from utils.confusion_matrix import calculate_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.kNN import kNN


def mean_words_amount_per_rating(df: pd.DataFrame):
    for rating in df["Star Rating"].unique():
        mean_word_count = df[df["Star Rating"] == rating]["wordcount"].mean()
        print(f"Star Rating {rating}: Average Word Count = {mean_word_count}")
    print("--------------------")


# def fill_missing_vals(df: pd.DataFrame):
#

def print_results(k, conf_matrix, per_label_conf_matrix):
    print(f"Results for k = {k}:")
    print("Confusion Matrix:")
    print(conf_matrix)

    # print("Per Label Confusion Matrix:")
    # print(per_label_conf_matrix)

    metrics_result = metrics(per_label_conf_matrix)
    print("Precision:")
    print(
        f"positive: {metrics_result['positive']['Precision']}, negative: {metrics_result['negative']['Precision']}")


def run_ej2():
    df = pd.read_csv("./data/reviews_sentiment.csv", delimiter=';', encoding='utf-8')
    print(f"Original Length: {len(df)}")
    df = df.drop_duplicates()
    print(f"Length with no duplicates: {len(df)}")
    # comment one of  the following lines
    df = df.dropna(subset=["wordcount", "titleSentiment", "textSentiment", "Star Rating", "sentimentValue"])
    # fill_missing_vals(df)
    print("--------------------")

    mean_words_amount_per_rating(df)
    df.drop(columns=["Review Title", "Review Text", "textSentiment"], inplace=True)
    k = 3

    train_df, test_df = k_fold_split(df, k)
    class_labels = np.array(['positive', 'negative'])

    print("kNN")
    for k in range(1, 42, 2):
        res = kNN(train_df, test_df, "titleSentiment", k, False)
        print(res)
        conf_matrix = calculate_confusion_matrix(class_labels, res['predictions'], res['titleSentiment'])
        per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_matrix)
        print_results(k, conf_matrix, per_label_conf_matrix)

    print("--------------------")
    print("weighted kNN")
    for k in range(1, 42, 2):
        res = kNN(train_df, test_df, "titleSentiment", k, True)
        conf_matrix = calculate_confusion_matrix(class_labels, res['predictions'], res['titleSentiment'])
        per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_matrix)
        print_results(k, conf_matrix, per_label_conf_matrix)