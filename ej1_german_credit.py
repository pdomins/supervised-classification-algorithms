import pandas as pd
import numpy as np
from utils.data_split import k_fold_split
from utils.decision_tree import id3


def discretize_variables(df: pd.DataFrame, var: str, bins_amount: int):
    min_val = df[var].min()
    max_val = df[var].max()
    bin_width = (max_val - min_val) / bins_amount
    bin_edges = [min_val + i * bin_width for i in range(bins_amount + 1)]
    bin_labels = [i for i in range(bins_amount)]
    df[var] = pd.cut(df[var], bins=bin_edges, labels=bin_labels, include_lowest=True)


def run_ej1():
    df = pd.read_csv("./data/german_credit.csv", delimiter=',', encoding='utf-8')
    discretize_variables(df, 'Credit Amount', 5)
    discretize_variables(df, 'Duration of Credit (month)', 4)
    discretize_variables(df, 'Age (years)', 3)

    train_df, test_df = k_fold_split(df, k=3)
    dec_tree          = id3(train_df, "Creditability")
    predicted_column  = test_df.apply(dec_tree.predict, axis=1)

    test_df.insert(loc=1, column="Creditability (predicted)", value=predicted_column)
    test_df.to_csv("ej1_german_credit_prediction.csv", sep=";")

    correct_predictions  = test_df[test_df['Creditability (predicted)'] == test_df['Creditability']].shape[0]
    incorrect_prediction = test_df.shape[0] - correct_predictions
    print(train_df)
    print(test_df)
    print({
        "test":{
            "correct"   : correct_predictions,
            "incorrect" : incorrect_prediction
        }
    })
