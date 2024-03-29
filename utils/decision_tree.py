from typing               import Any
from dataclasses          import dataclass, replace
from collections          import deque
from utils.dec_tree_utils import AttrCat, categorize_attrs_by_vals_from_df, apply_cats_to_df, gain, pre_pruning_from_dict, PrePruning, DecisionTreeProperties
import pandas as pd

@dataclass
class AttrNode:
    attr       : str
    values     : dict[Any, 'ValueNode']

@dataclass
class LeafNode:
    out_label  : Any

@dataclass
class ValueNode:
    value      : AttrCat
    next_level : AttrNode | LeafNode
    train_df   : pd.DataFrame = None

def decide4attr(attr_node : AttrNode, sample : pd.Series) -> ValueNode:
    for value_key in attr_node.values.keys():
        curr_value_node = attr_node.values[value_key]
        if curr_value_node.value.slct_f(sample):
            return curr_value_node


class DecisionTree:

    def __init__(self, root : AttrNode, out_attr : str, tree_props : DecisionTreeProperties = None):
        self.root     = root
        self.out_attr = out_attr
        if tree_props is not None:
            self.tree_props = replace(tree_props)
        
    def predict(self, test_sample : pd.Series) -> Any:
        curr_node         = self.root
        predict_decisions = []

        while not isinstance(curr_node, LeafNode):
            curr_val_node = decide4attr(curr_node, test_sample)

            predict_decisions.append({
                curr_node.attr : curr_val_node.value.cat_label
            })

            curr_node     = curr_val_node.next_level
        
        self.__last_predict_decisions__ = predict_decisions
        return curr_node.out_label
    
    def __as_dict__(self) -> dict:
        node_queue = deque()

        root_repr_node  = dict()
        root_node       = self.root

        if isinstance(root_node, LeafNode):
            root_repr_node = { self.out_attr : root_node.out_label }
        else:
            node_queue.append((root_node, root_repr_node))

        while node_queue:
            curr_node, repr_node = node_queue.popleft()

            repr_node[curr_node.attr] = dict()
            repr_node = repr_node[curr_node.attr]

            for val_key in curr_node.values.keys():
                curr_val_node        = curr_node.values[val_key]
                cat_label            = curr_val_node.value.cat_label

                next_level = curr_val_node.next_level
                if isinstance(next_level, LeafNode):
                    repr_node[cat_label] = { self.out_attr : next_level.out_label }
                else:
                    repr_node[cat_label] = dict()
                    node_queue.append((curr_val_node.next_level, repr_node[cat_label]))
        
        return root_repr_node
    
    def __repr__(self) -> str:
        dec_tree_as_dict = self.__as_dict__()
        return dec_tree_as_dict.__repr__()



def __max_gain_attr__(df : pd.DataFrame, out_col : str, attrs_by_priority : list[str], cats4attrs : dict[str, list[AttrCat]]) -> str:
    max_g_attr = None
    max_g      = None
    for attr in attrs_by_priority:
        g = gain(df, out_col, cats4attr=cats4attrs[attr])

        if max_g is None or g > max_g:
            max_g_attr = attr
            max_g      = g

    return max_g_attr

def __build_branch__(df               : pd.DataFrame, 
                     out_col          : str, 
                     remaining_attrs_by_priority : list[str], 
                     cats4attrs       : dict[str, list[AttrCat]], 
                     max_g_attr       : str, 
                     depth            : int,
                     tree_props       : DecisionTreeProperties,
                     pre_pruning      : PrePruning,
                     save_decision_df : bool = False) -> AttrNode:
    branches = dict()
    cats_dfs_dict = apply_cats_to_df(df, cats4attrs[max_g_attr])
    for cat_key in cats_dfs_dict.keys():
        curr_cat          = next(filter(lambda attr_cat : attr_cat.cat_label == cat_key, cats4attrs[max_g_attr]))
        curr_cat_df       = cats_dfs_dict[cat_key]
        child_branch      = __id3_dfs__(curr_cat_df, out_col, remaining_attrs_by_priority, cats4attrs, depth + 1, tree_props, pre_pruning, df, save_decision_df=save_decision_df)
        
        value_node        = ValueNode(curr_cat, child_branch) if not save_decision_df \
                       else ValueNode(curr_cat, child_branch, curr_cat_df)
        
        branches[cat_key] = value_node
        tree_props.val_node_count += 1

    branch = AttrNode(max_g_attr, branches)
    tree_props.attr_node_count += 1
    return branch

def __is_base_case__(df : pd.DataFrame, out_col : str, remaining_attrs_by_priority : list[str], depth : int, pre_prunning : PrePruning) -> int:
    
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
    
    # Base case #4 : Max depth is set and we will exceeded it
    if pre_prunning.max_depth is not None and depth >= pre_prunning.max_depth:
        return 4
    
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

# Base case #4 : Max depth is set and we will exceeded it
def __build_leaf_node_for_max_depth_exceeded__(df : pd.DataFrame, out_col : str):
    label_mode = __get_mode_out_label__(df, out_col)
    return LeafNode(label_mode)

base_case_dict = {
    1 : lambda df, out_col, _         : __build_leaf_node_for_same_out_label__(df, out_col),
    2 : lambda _,  out_col, parent_df : __build_leaf_node_for_no_samples__(parent_df, out_col),
    3 : lambda df, out_col, _         : __build_leaf_node_for_no_attrs_remain__(df, out_col),
    4 : lambda df, out_col, _         : __build_leaf_node_for_max_depth_exceeded__(df, out_col)
}

def __build_leaf_node__(base_case : int, df : pd.DataFrame, out_col : str, parent_df : pd.DataFrame) -> LeafNode:
    return base_case_dict[base_case](df, out_col, parent_df)

def __id3_dfs__(df      : pd.DataFrame, 
                out_col : str, 
                remaining_attrs_by_priority : list[str], 
                cats4attrs       : dict[str, list[AttrCat]], 
                depth            : int,
                tree_props       : DecisionTreeProperties,
                pre_pruning      : PrePruning,
                parent_df        : pd.DataFrame = None, 
                save_decision_df : bool = False) -> AttrNode | LeafNode:
    base_case = __is_base_case__(df, out_col, remaining_attrs_by_priority, depth, pre_pruning)
    if base_case is not None :
        leaf_node = __build_leaf_node__(base_case, df, out_col, parent_df)
        tree_props.leaf_node_count += 1
        tree_props.depth = max(tree_props.depth, depth + 1)
        return leaf_node

    max_g_attr                  = __max_gain_attr__(df, out_col, remaining_attrs_by_priority, cats4attrs)
    remaining_attrs_by_priority = list(filter(lambda attr : attr != max_g_attr, remaining_attrs_by_priority))
    tree_props.attrs.add(max_g_attr)
    
    branch = __build_branch__(df, out_col, remaining_attrs_by_priority, cats4attrs, max_g_attr, depth, tree_props, pre_pruning, save_decision_df=save_decision_df)
    return branch

def id3(df                : pd.DataFrame, 
        out_col           : str, 
        attrs_by_priority : list[str] = None, 
        attrs_vals        : dict[str, list[Any]] = None, 
        cats4attrs        : dict[str, list[AttrCat]] = None, 
        save_decision_df  : bool = False,
        pre_pruning       : dict[str, Any] = None) -> DecisionTree:

    if df.empty:
        raise ValueError("cannot create DecisionTree from empty DataFrame.")

    if attrs_by_priority is None:
        no_out_df         = df.drop(columns=[out_col])
        attrs_by_priority = no_out_df.columns

    cats4attrs = categorize_attrs_by_vals_from_df(df, attrs_by_priority, attrs_vals, cats4attrs)

    if pre_pruning is None:
        pre_pruning = dict()

    pre_pruning_data = pre_pruning_from_dict(pre_pruning)

    tree_props = DecisionTreeProperties(depth=0, attr_node_count=0, val_node_count=0, leaf_node_count=0)

    root = __id3_dfs__(df, out_col, attrs_by_priority, cats4attrs, 0, tree_props, pre_pruning_data, save_decision_df=save_decision_df)

    return DecisionTree(root, out_col, tree_props)

def decision_trees_over_possible_depths(
        df                : pd.DataFrame, 
        out_col           : str, 
        depth_limit       : tuple[int, int] = None, 
        attrs_by_priority : list[str] = None, 
        attrs_vals        : dict[str, list[Any]] = None, 
        cats4attrs        : dict[str, list[AttrCat]] = None, 
        save_decision_df  : bool = False,
        pre_pruning       : dict[str, Any] = None) -> dict[int, DecisionTree]:
    
    if pre_pruning is not None:
        pre_pruning = pre_pruning.copy()
        pre_pruning.pop("max_depth", None)
    else:
        pre_pruning = dict()

    min_depth = 0
    max_depth = None

    if depth_limit is not None:
        min_depth = depth_limit[0] if depth_limit[0] is not None \
            else    0
        max_depth = depth_limit[1]

    if max_depth is not None and min_depth > max_depth:
        return []
    
    trees_by_depth = dict()
    
    if max_depth is None:
        max_depth_dec_tree = id3(df, out_col, attrs_by_priority, attrs_vals, cats4attrs, save_decision_df, pre_pruning)
        max_depth = max_depth_dec_tree.tree_props.depth
        dec_tree_node_count = max_depth_dec_tree.tree_props.val_node_count
        trees_by_depth[dec_tree_node_count] = max_depth_dec_tree

    print(min_depth, max_depth)
    for depth in range(max_depth-1, min_depth-1, -1):
        print(depth)
        pre_pruning["max_depth"] = depth
        dec_tree = id3(df, out_col, attrs_by_priority, attrs_vals, cats4attrs, save_decision_df, pre_pruning)

        dec_tree_node_count = dec_tree.tree_props.val_node_count
        if dec_tree_node_count not in trees_by_depth:
            trees_by_depth[dec_tree_node_count] = dec_tree
    
    return trees_by_depth