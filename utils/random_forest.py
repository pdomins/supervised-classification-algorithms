import numpy as np
import pandas as pd

from utils.decision_tree import id3
from utils.decision_tree import DecisionTree
from utils.avg_utils import calculate_average


class RandomForest:

    def __init__(self, train_set: pd.DataFrame, bag_count: int, sample_size: int, attrs_vals: list, max_depth: int | None = None):
        self.bag_count : int = bag_count
        self.sample_size : int = sample_size
        self.bags = dict()
        self.models : dict[DecisionTree] = dict()
        for idx in range(bag_count):
            self.bags[idx] = train_set.sample(sample_size, replace=True)
            if (max_depth != None):
                pre_pruning = dict()
                pre_pruning["max_depth"] = max_depth
                self.models[idx] = id3(self.bags[idx], "Creditability", attrs_vals=attrs_vals, pre_pruning=pre_pruning)
            else:
                self.models[idx] = id3(self.bags[idx], "Creditability", attrs_vals=attrs_vals)

    def predict(self, test_sample: pd.Series):
        predictions = []
        for idx in range(self.bag_count):
            dec_tree : DecisionTree = self.models[idx]
            prediction = dec_tree.predict(test_sample)
            predictions.append(prediction)
        return calculate_average(predictions)
    
    def print(self):
        print(f'bag_count: {self.bag_count}, sample_size: {self.sample_size}')
        print(f'models: {self.models}')
