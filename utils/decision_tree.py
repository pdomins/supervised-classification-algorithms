from typing         import Any
from dataclasses    import dataclass
from dec_tree_utils import AttrCat, categorize_attrs_by_vals_from_df, apply_cats_to_df, gain
import pandas as pd

@dataclass
class AttrNode:
    attr       : str
    values     : dict[Any, 'ValueNode']

@dataclass
class ValueNode:
    value      : Any
    next_level : AttrNode | 'LeafNode'

@dataclass
class LeafNode:
    out_label  : Any


class DecisionTree:

    def __init__(self, trunk : AttrNode):
        self.trunk = trunk
        
    def predict(self, test_sample : pd.Series):
        # TODO
        pass


def __max_gain_attr__(df : pd.DataFrame, out_col : str, attrs_by_priority : list[str], cats4attrs : dict[str, list[AttrCat]]) -> str:
    max_g_attr = None
    max_g      = None
    for attr in attrs_by_priority:
        g = gain(df, out_col, cats4attr=cats4attrs[attr])

        if max_g is None or g > max_g:
            max_g_attr = attr
            max_g      = g

    return max_g_attr

def __build_branch__(df : pd.DataFrame, out_col : str, remaining_attrs_by_priority : list[str], cats4attrs : dict[str, list[AttrCat]], max_g_attr : str) -> dict[Any, 'ValueNode']:
    branch        = dict()
    cats_dfs_dict = apply_cats_to_df(df, cats4attrs[max_g_attr])
    for cat_key in cats_dfs_dict.keys():
        curr_cat_df       = cats_dfs_dict[cat_key]
        child_branch      = __id3_dfs__(curr_cat_df, out_col, remaining_attrs_by_priority, cats4attrs, df)
        value_node        = ValueNode(cat_key, child_branch)
        branch[cat_key]   = value_node
    return branch

def __is_base_case__(df : pd.DataFrame, out_col : str, remaining_attrs_by_priority : list[str]) -> int:
    
    # Base case #1 : Same label for all samples
    diff_out_label_count = df[out_col].unique().shape[0]
    if diff_out_label_count == 1:
        return 1

    # Base case #2 : No samples
    if df.empty: 
        return 2
    
    # Base case #3 : Used up every attr
    remaining_attrs_count = len(remaining_attrs_by_priority)
    if remaining_attrs_count == 0:
        return 3
    
    return None

def __get_mode_out_label__(df : pd.DataFrame, out_col : str) -> str:
    value_counts = df[out_col].value_counts()
    label_mode   = value_counts.idxmax()
    return label_mode

def __get_only_out_label__(df : pd.DataFrame, out_col : str) -> str:
    out_labels = df[out_col].unique()
    if len(out_labels) != 1:
        raise ValueError("unexpected number of out labels for same output label base case.")
    
    out_label  = out_labels[0]
    return out_label

# Base case #1 : Same label for all samples
def __build_leaf_node_for_same_out_label__(df : pd.DataFrame, out_col : str):
    only_out_label = __get_only_out_label__(df, out_col)
    return LeafNode(only_out_label)

# Base case #2 : No samples
def __build_leaf_node_for_no_samples__(parent_df : pd.DataFrame, out_col : str):
    parent_label_mode = __get_mode_out_label__(parent_df, out_col)
    return LeafNode(parent_label_mode)

# Base case #3 : Used up every attr
def __build_leaf_node_for_no_attrs_remain__(df : pd.DataFrame, out_col : str):
    label_mode = __get_mode_out_label__(df, out_col)
    return LeafNode(label_mode)

base_case_dict = {
    1 : lambda df, out_col, _         : __build_leaf_node_for_same_out_label__(df, out_col),
    2 : lambda _,  out_col, parent_df : __build_leaf_node_for_no_samples__(parent_df, out_col),
    3 : lambda df, out_col, _         : __build_leaf_node_for_no_attrs_remain__(df, out_col)
}

def __build_leaf_node__(base_case : int, df : pd.DataFrame, out_col : str, parent_df : pd.DataFrame) -> LeafNode:
    return base_case_dict[base_case](df, out_col, parent_df)

def __id3_dfs__(df : pd.DataFrame, out_col : str, remaining_attrs_by_priority : list[str], cats4attrs : dict[str, list[AttrCat]], parent_df : pd.DataFrame) -> AttrNode | LeafNode:
    base_case     = __is_base_case__(df, out_col, remaining_attrs_by_priority)
    if base_case is not None :
        return __build_leaf_node__(base_case, df, out_col, parent_df)

    max_g_attr                  = __max_gain_attr__(df, out_col, remaining_attrs_by_priority, cats4attrs)
    remaining_attrs_by_priority = list(filter(lambda attr : attr != max_g_attr, remaining_attrs_by_priority))
    
    branch     = __build_branch__(df, out_col, remaining_attrs_by_priority, cats4attrs, max_g_attr)
    attr_node  = AttrNode(max_g_attr, branch)
    return attr_node

def id3(df : pd.DataFrame, out_col : str, attrs_by_priority : list[str] = None, attrs_vals : dict[str, list[Any]] = None, cats4attrs : dict[str, list[AttrCat]] = None) -> DecisionTree:
    
    if attrs_by_priority is None:
        no_out_df         = df.drop(columns=[out_col])
        attrs_by_priority = no_out_df.columns

    cats4attrs = categorize_attrs_by_vals_from_df(df, attrs_by_priority, attrs_vals, cats4attrs)

    trunk = __id3_dfs__(df, out_col, attrs_by_priority, cats4attrs)

    return DecisionTree(trunk)