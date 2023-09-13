from typing         import Any
from dataclasses    import dataclass
from dec_tree_utils import AttrCat, categorize_attrs_by_vals_from_df, gain
import pandas as pd

class DecisionTree:

    def __init__(self):
        # TODO
        pass
        
    def predict(self, test_sample : pd.Series):
        # TODO
        pass

@dataclass
class LeafNode:
    value    : Any

@dataclass
class TreeNode:
    value    : Any
    children : dict[Any, 'TreeNode' | LeafNode]

@dataclass
class AttrNode(TreeNode):
    value    : str
    children : dict[str, TreeNode | LeafNode]

@dataclass
class ValueNode(TreeNode):
    pass

def __max_gain_attr__(df : pd.DataFrame, out_col : str, attrs : list[str], cats4attrs : dict[str, list[str]]) -> str:
    max_g_attr = None
    max_g      = None
    for attr in attrs:
        g = gain(df, out_col, cats4attr=cats4attrs[attr])

        if max_g is None or g > max_g:
            max_g_attr = attr
            max_g      = g

    return max_g_attr

def __build_branch__():
    pass

def __id3_dfs__(df : pd.DataFrame, out_col : str, attrs : list[str], cats4attrs : dict[str, list[AttrCat]]) -> TreeNode:
    max_g_attr = __max_gain_attr__(df, out_col, attrs, cats4attrs)
    value_node = ValueNode()
    attr_node  = AttrNode(max_g_attr, None)

def id3(df : pd.DataFrame, out_col : str, attrs : list[str] = None, attrs_vals : dict[str, list[Any]] = None, cats4attrs : dict[str, list[AttrCat]] = None) -> DecisionTree:
    
    if attrs is None:
        no_out_df = df.drop(columns=[out_col])
        attrs     = no_out_df.columns

    cats4attrs = categorize_attrs_by_vals_from_df(df, attrs, attrs_vals, cats4attrs)

    __id3_dfs__(df, out_col, attrs, cats4attrs)