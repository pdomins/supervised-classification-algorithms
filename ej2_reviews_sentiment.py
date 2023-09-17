import pandas as pd
import numpy as np
from utils.data_split import k_fold_split
from utils.confusion_matrix import calculate_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.kNN import kNN


def title_sentiment_stats(df: pd.DataFrame):
    value_counts = df['titleSentiment'].value_counts()
    print("Title Sentiment")
    print(f"positive: {value_counts['positive']}, negative: {value_counts['negative']}")
    print("--------------------")


def mean_words_amount_per_rating(df: pd.DataFrame):
    for rating in df["Star Rating"].unique():
        mean_word_count = df[df["Star Rating"] == rating]["wordcount"].mean()
        print(f"Star Rating {rating}: Average Word Count = {mean_word_count}")
    print("--------------------")


def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)


def normalize_df(df: pd.DataFrame):
    df['wordcount'] = normalize_column(df['wordcount'])
    df['titleSentiment'] = df['titleSentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    df['sentimentValue'] = normalize_column(df['sentimentValue'])
    return df


def handle_n_a(df: pd.DataFrame, drop: bool = True):
    if drop:
        df = df.dropna(subset=["wordcount", "titleSentiment", "textSentiment", "Star Rating", "sentimentValue"])
    return df


def print_results(k, conf_matrix, per_label_conf_matrix):
    print(f"\nResults for k = {k}:")
    print("Confusion Matrix:")
    print(conf_matrix)

    # print("Per Label Confusion Matrix:")
    # print(per_label_conf_matrix)

    metrics_result = metrics(per_label_conf_matrix)
    for key, value in metrics_result.items():
        precision = value['Precision']
        print(f'Key {key}: Precision = {precision}')


def run_ej2():
    df = pd.read_csv("./data/reviews_sentiment.csv", delimiter=';', encoding='utf-8')
    print(f"Original Length: {len(df)}")

    df = df.drop_duplicates()
    print(f"Length with no duplicates: {len(df)}")

    df = handle_n_a(df, drop=True)
    print("--------------------")
    title_sentiment_stats(df)
    mean_words_amount_per_rating(df)
    df.drop(columns=["Review Title", "Review Text", "textSentiment"], inplace=True)

    df = normalize_df(df)
    print(df)
    # k split
    k = 3
    train_df, test_df = k_fold_split(df, k)

    # kNN
    to_predict = "Star Rating"
    # to_predict = "titleSentiment"
    class_labels = np.array(df[to_predict].unique())
    start = 1
    stop = 42
    step = 2

    print("kNN")
    for k in range(start, stop, step):
        res = kNN(train_df, test_df, to_predict, k, False)
        conf_matrix = calculate_confusion_matrix(class_labels, res['predictions'], res[to_predict])
        per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_matrix)
        print_results(k, conf_matrix, per_label_conf_matrix)

    print("--------------------")
    print("weighted kNN")
    for k in range(start, stop, step):
        res = kNN(train_df, test_df, to_predict, k, True)
        conf_matrix = calculate_confusion_matrix(class_labels, res['predictions'], res[to_predict])
        per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_matrix)
        print_results(k, conf_matrix, per_label_conf_matrix)
