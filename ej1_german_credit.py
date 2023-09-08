import pandas as pd
import numpy as np
from utils.data_split import k_fold_split


def run_ej1():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    random_generator = np.random.default_rng(seed=42)
    k = 3

    train_df, test_df = k_fold_split(df, k, random_generator)
    print(train_df)
    print(test_df)

