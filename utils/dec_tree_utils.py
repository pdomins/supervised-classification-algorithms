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

def __attr_val_gain__(Sv : pd.DataFrame, out_label : str, S_count : int) -> float:
    Sv_count = Sv.shape[0]
    h_Sv     = shannon_entropy(Sv[out_label])
    
    Sv_gain  = (Sv_count / S_count) * h_Sv
    return Sv_gain

def __attr_gain__(S : pd.DataFrame, attr : str, out_label : str, S_count : int) -> float:
    attr_vals = S[attr].unique()
    """
        attr_vals example:

        array([1, 0])
    """

    attr_gain = 0
    for val in attr_vals:
        Sv         = S[S[attr] == val]
        Sv_gain    = __attr_val_gain__(Sv, out_label, S_count)
        attr_gain += Sv_gain
    return attr_gain

def gain(S : pd.DataFrame, attr : str, out_label : str) -> float:
    S_count = S.shape[0]
    h_S     = shannon_entropy(S[out_label])
    
    attr_gain = __attr_gain__(S, attr, out_label, S_count)
    return h_S - attr_gain