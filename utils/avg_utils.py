from collections import Counter


def __calculate_average_string(class_predictions: list):
    class_string_counts = Counter(
        class_predictions)  # [('string1', maxcount), ('string2', count), ... , ('stringN, mincount)]
    most_common_class = class_string_counts.most_common(1)[0][0]
    return most_common_class


def calculate_average(predictions: list):
    if isinstance(predictions[0], (int, float)):
        predictions = list(map(str, predictions))
    return __calculate_average_string(predictions)
