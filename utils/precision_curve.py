import numpy  as np
import pandas as pd
from utils.decision_tree import DecisionTree, decision_trees_over_possible_depths
from utils.dec_tree_utils import AttrCat
from utils.confusion_matrix import metrics, calculate_per_label_confusion_matrix
from typing import Any

from utils.random_forest import RandomForest

def precisions_over_possible_depth(
        train_df            : pd.DataFrame, 
        test_df             : pd.DataFrame,
        possible_out_labels : list[str],
        out_col             : str, 
        attrs_vals          : dict[str, list[Any]], 
        depth_limit         : tuple[int, int] = None,
        attrs_by_priority   : list[str] = None,
        cats4attrs          : dict[str, list[AttrCat]] = None, 
        save_decision_df    : bool = False,
        pre_pruning         : dict[str, Any] = None) -> dict[Any, dict[int, float]]:
    
    dec_trees_dict = decision_trees_over_possible_depths(train_df, out_col, depth_limit, attrs_by_priority, attrs_vals, cats4attrs, save_decision_df, pre_pruning)
    
    precision_by_label_by_depth = dict()
    for label in possible_out_labels:
        precision_by_label_by_depth[label] = dict()
    
    for depth in dec_trees_dict.keys():
        curr_dec_tree = dec_trees_dict[depth]

        predicted = test_df.apply(curr_dec_tree.predict, axis=1).to_dict()
        expected  = test_df[out_col].to_dict()

        per_label_conf_mats = calculate_per_label_confusion_matrix(possible_out_labels, predicted, expected)

        metric_dict = metrics(per_label_conf_mats)
        for label in metric_dict.keys():
            precision_by_label_by_depth[label][depth] = metric_dict[label]["Precision"]

    return precision_by_label_by_depth


def precisions_over_possible_depth_forest(
        train_df            : pd.DataFrame, 
        test_df             : pd.DataFrame,
        possible_out_labels : list[str],
        out_col             : str, 
        attrs_vals          : dict[str, list[Any]]) -> dict[Any, dict[int, float]]:
    
    rf_dict = dict()
    for i in range(20):
        rf_dict[i] = RandomForest(train_df, bag_count=40, sample_size=200, attrs_vals=attrs_vals, max_depth=i+1)
    
    precision_by_label_by_node_count = dict()
    for label in possible_out_labels:
        precision_by_label_by_node_count[label] = dict()
    
    for node_count in rf_dict.keys():
        curr_rf : RandomForest = rf_dict[node_count]

        print(f"Predicting {node_count}")
        predicted = test_df.apply(curr_rf.predict, axis=1).to_dict()
        expected  = test_df[out_col].to_dict()

        per_label_conf_mats = calculate_per_label_confusion_matrix(possible_out_labels, predicted, expected)

        metric_dict = metrics(per_label_conf_mats)
        for label in metric_dict.keys():
            precision_by_label_by_node_count[label][node_count] = metric_dict[label]["Precision"]

    return precision_by_label_by_node_count



def precisions_over_count_forest(
        train_df            : pd.DataFrame, 
        test_df             : pd.DataFrame,
        possible_out_labels : list[str],
        out_col             : str, 
        attrs_vals          : dict[str, list[Any]]) -> dict[Any, dict[int, float]]:
    
    rf_dict = dict()
    for i in range(20):
        curr_rf = RandomForest(train_df, bag_count=40, sample_size=200, attrs_vals=attrs_vals, max_depth=i+1)
        # sorted_model_list = sorted(, key=lambda dt: dt., reverse=True)
        node_count = curr_rf.models[0].tree_props.val_node_count
        rf_dict[node_count] = curr_rf
            
    precision_by_label_by_node_count = dict()
    for label in possible_out_labels:
        precision_by_label_by_node_count[label] = dict()
    
    for node_count in rf_dict.keys():
        curr_rf : RandomForest = rf_dict[node_count]

        print(f"Predicting {node_count}")
        predicted = test_df.apply(curr_rf.predict, axis=1).to_dict()
        expected  = test_df[out_col].to_dict()

        per_label_conf_mats = calculate_per_label_confusion_matrix(possible_out_labels, predicted, expected)

        metric_dict = metrics(per_label_conf_mats)
        for label in metric_dict.keys():
            precision_by_label_by_node_count[label][node_count] = metric_dict[label]["Precision"]

    return precision_by_label_by_node_count