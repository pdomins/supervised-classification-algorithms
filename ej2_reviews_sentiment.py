import pandas as pd
import numpy as np
from utils.data_split import k_fold_split
from utils.kNN import kNN


def mean_words_amount_per_rating(df: pd.DataFrame):
    for rating in df["Star Rating"].unique():
        mean_word_count = df[df["Star Rating"] == rating]["wordcount"].mean()
        print(f"Star Rating {rating}: Average Word Count = {mean_word_count}")
    print("--------------------")


# def fill_missing_vals(df: pd.DataFrame):
#


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
    k = 10
    kNN(train_df, test_df, "titleSentiment", k, False)
