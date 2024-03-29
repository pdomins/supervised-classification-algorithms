import numpy as np
from collections import defaultdict


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_distances(train_df, test_row, attr_to_predict, k):
    distances = []
    for idx, train_row in train_df.iterrows():
        dist = euclidean_distance(train_row.drop(attr_to_predict), test_row.drop(attr_to_predict))
        distances.append((idx, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:k]


def find_max_prediction(predictions):
    max_key = max(predictions, key=lambda k: predictions[k])
    max_value = predictions[max_key]
    if list(predictions.values()).count(max_value) > 1:
        return -1  # Return -1 in case of a draw
    else:
        return max_key


def get_predictions(distances, train_df, is_weighted, attr_to_predict):
    predictions = defaultdict(float)
    for idx, dist in distances:
        text_sentiment = train_df.loc[idx, attr_to_predict]
        if is_weighted:
            predictions[text_sentiment] += 1 / ((dist + 1e-6) ** 2)  ##TODO FIX ME ASAP
        else:
            predictions[text_sentiment] += 1
    return find_max_prediction(predictions)


def get_prediction_for_test_row(train_df, test_row, attr_to_predict, k, is_weighted):
    distances = calculate_distances(train_df, test_row, attr_to_predict, k)
    prediction = get_predictions(distances, train_df, is_weighted, attr_to_predict)
    new_k = k + 1
    while prediction == -1:
        distances = calculate_distances(train_df, test_row, attr_to_predict, new_k)
        prediction = get_predictions(distances, train_df, is_weighted, attr_to_predict)
        new_k += 1
    return prediction


def kNN(train_df, test_df, attr_to_predict, k, is_weighted=False):
    if k > len(train_df):
        raise ValueError(f"k should be smaller than the total amount of points {len(train_df)}")

    test_df_copy = test_df.copy()
    predictions = [get_prediction_for_test_row(train_df, test_row, attr_to_predict, k, is_weighted)
                   for _, test_row in test_df_copy.iterrows()]

    test_df_copy['predictions'] = predictions
    return test_df_copy
