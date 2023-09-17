import pandas as pd
import numpy as np
from utils.data_split import k_fold_split
from utils.confusion_matrix import calculate_confusion_matrix, \
    calculate_per_label_confusion_matrix_from_confusion_matrix, metrics
from utils.kNN import kNN
from utils.kNN_utils import get_best_k, normalize_column


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


def normalize_df(df: pd.DataFrame):
    df['wordcount'] = normalize_column(df['wordcount'])
    df['titleSentiment'] = df['titleSentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    df['sentimentValue'] = normalize_column(df['sentimentValue'])
    return df


def set_title_sentiment(row):
    if pd.isna(row['titleSentiment']):
        if pd.notna(row['Star Rating']) and row['Star Rating'] >= 3:
            return 'positive'
        else:
            return 'negative'
    else:
        return row['titleSentiment']


def handle_n_a(df: pd.DataFrame, drop_missing_values: bool = True):
    if drop_missing_values:
        df = df.dropna(subset=["wordcount", "titleSentiment", "textSentiment", "Star Rating", "sentimentValue"])
    else:
        df = df.dropna(subset=["wordcount", "textSentiment", "Star Rating", "sentimentValue"])
        df['titleSentiment'] = df.apply(set_title_sentiment, axis=1)
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

    df = handle_n_a(df, drop_missing_values=False)
    print(f"Length after handling NaN: {len(df)}")

    print("--------------------")
    df.drop(columns=["Review Title", "Review Text", "textSentiment"], inplace=True)

    # stats
    title_sentiment_stats(df)
    mean_words_amount_per_rating(df)

    # normalize
    df = normalize_df(df)

    # k split
    k_fold = 4
    train_df, test_df = k_fold_split(df, k_fold)

    # kNN
    to_predict = "Star Rating"
    class_labels = np.array(df[to_predict].unique())
    # kNN

    k = get_best_k(train_df, test_df, to_predict, class_labels, False)  # get most precise k for dataframe

    res = kNN(train_df, test_df, to_predict, k, False)
    conf_matrix = calculate_confusion_matrix(class_labels, res['predictions'], res[to_predict])
    per_label_conf_matrix = calculate_per_label_confusion_matrix_from_confusion_matrix(conf_matrix)
    print_results(k, conf_matrix, per_label_conf_matrix)

    # weighted kNN
    # k = get_best_k(train_df, test_df, to_predict, class_labels, True)
