import pandas as pd
import numpy as np
from utils.data_split import k_fold_split


def mean_words_amount_per_rating(df: pd.DataFrame):
    for rating in df["Star Rating"].unique():
        mean_word_count = df[df["Star Rating"] == rating]["wordcount"].mean()
        print(f"Star Rating {rating}: Average Word Count = {mean_word_count}")


def run_ej2():
    df = pd.read_csv("./data/reviews_sentiment.csv", delimiter=';', encoding='utf-8')
    mean_words_amount_per_rating(df)

    random_generator = np.random.default_rng(seed=42)
    k = 3

    train_df, test_df = k_fold_split(df, k, random_generator)

    print(train_df)
    print(test_df)
