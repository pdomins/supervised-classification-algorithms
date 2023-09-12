import numpy as np
import pandas as pd

from decision_tree import DecisionTree
from avg_utils import calculate_average

# TODO: Replace
# from id3 import id3
def __id3(dataset : pd.DataFrame) -> DecisionTree:
    return DecisionTree()

class RandomForest:

    def __init__(self, train_set : pd.DataFrame, bag_count: int, sample_size: int):
        self.bag_count = bag_count
        self.sample_size = sample_size
        self.bags = np.empty(bag_count)
        self.models = np.empty(bag_count)
        for idx in range(bag_count):
            self.bags[idx] = train_set.sample(sample_size, replace=True)
            self.models[idx] = __id3(self.bags[idx])

    def predict(self, test_sample : pd.Series):
        predictions = np.empty(self.bag_count)
        for model in self.models:
            np.append(predictions, model.predict(test_sample))
        return calculate_average(predictions)