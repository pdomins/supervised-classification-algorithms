import pandas as pd
import numpy as np
from collections import defaultdict


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def record_euclidean_aggregate(df_row, to_predict_row, attr_to_predict):
    distance = 0
    for column_name, value in df_row.items():
        if column_name != attr_to_predict:
            distance += euclidean_distance(float(value), float(to_predict_row[column_name]))
    return distance


def get_predictions(data_dict, is_weighted, dict_key):
    predictions = defaultdict(float)

    for values in data_dict.values():
        text_sentiment = values[dict_key]
        distance = values['distance']

        if is_weighted:
            predictions[text_sentiment] = 1 / distance ** 2 if distance != 0 else 0
        else:
            predictions[text_sentiment] += 1

    return max(predictions, key=lambda k: predictions[k])


def kNN(df: pd.DataFrame, test_df: pd.DataFrame, attr_to_predict: str, k: int, is_weighted: bool = False):
    if k > len(df):
        print(f"k should be smaller than the total amount of points {len(df)}")
        exit(1)

    predictions = []
    # for each value to predict
    original_df = df.copy()
    for _, to_predict_row in test_df.iterrows():
        df = original_df.copy()
        distances = {}
        for idx, df_row in df.iterrows():
            dist = record_euclidean_aggregate(df_row, to_predict_row, attr_to_predict)
            distances[idx] = {f'{attr_to_predict}': df_row[attr_to_predict], 'distance': dist}
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]['distance'], reverse=False)[
                                :k])  # reverse=False ascending order
        predictions.append(get_predictions(sorted_distances, is_weighted, attr_to_predict))

    test_df['predictions'] = predictions
    print(test_df)
