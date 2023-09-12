import pandas as pd
import math

def __label_shannon_entropy__(positive_cases : int, total_cases : int) -> float:
    if positive_cases == 0:
        return 0
    relative_probability = positive_cases / total_cases
    return - relative_probability * math.log2(relative_probability)

def shannon_entropy(X : pd.Series) -> float:
    label_count   = X.value_counts()
    """
        label_count example:

        Creditability
        1    700
        0    300
        Name: count, dtype: int64
    """

    total_samples = label_count.sum()
    """
        total_samples example:

        1000
    """

    h = 0
    for idx in label_count.index:
        """
            label_count.index example:

            Index([1, 0], dtype='int64', name='Creditability')
        """

        h += __label_shannon_entropy__(label_count[idx], total_samples)
    return h