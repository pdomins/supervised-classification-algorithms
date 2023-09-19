import numpy as np
import pandas as pd

from utils.decision_tree import id3
from utils.decision_tree import DecisionTree
from utils.avg_utils import calculate_average


class RandomForest:

    def __init__(self, train_set: pd.DataFrame, bag_count: int, sample_size: int):
        self.bag_count : int = bag_count
        self.sample_size : int = sample_size
        self.bags = {}
        self.models = {}
        for idx in range(bag_count):
            self.bags[idx] = train_set.sample(sample_size, replace=True)
            self.models[idx] = id3(self.bags[idx], "Creditability")

    def predict(self, test_sample: pd.Series):
        predictions = np.empty(self.bag_count)
        for idx in range(self.bag_count):
            np.append(predictions, self.models[idx].predict(test_sample))
        return calculate_average(predictions)
    
    def print(self):
        print(f'bag_count: {self.bag_count}, sample_size: {self.sample_size}')
        print(f'models: {self.models}')
