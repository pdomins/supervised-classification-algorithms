from dataclasses import dataclass, field
from typing import Callable, Any
import pandas as pd
import math


@dataclass
class AttrCat:
    cat_label: Any
    slct_f: Callable[[pd.DataFrame | pd.Series], pd.Series | bool]


def cat_value_counts(X: pd.DataFrame, cats4attr: list[AttrCat]) -> pd.Series:
    value_counts = dict()
    for cat in cats4attr:
        X_v = X[cat.slct_f(X)]

        value_counts[cat.cat_label] = X_v.count()
    return pd.Series(data=value_counts, name="count")


def __create_attr_cat__(cat_label: Any, df_slct_f: Callable[[pd.DataFrame], pd.Series],
                        s_slct_f: Callable[[pd.Series], bool]) -> AttrCat:
    def slct_f(df: pd.DataFrame | pd.Series) -> pd.Series | bool:
        if isinstance(df, pd.DataFrame):
            return df_slct_f(df)

        if isinstance(df, pd.Series):
            return s_slct_f(df)

        raise ValueError("df must be pandas DataFrame or Series")

    return AttrCat(cat_label, slct_f)


def categorize_attr(cat_defs: list[dict[str, Any]]) -> list[AttrCat]:
    cat_def_required_keys = set(["cat_label", "df_slct_f", "s_slct_f"])
    cats4attr = list()
    i = 0
    for cat_def in cat_defs:
        if not cat_def_required_keys.issubset(cat_def.keys()):
            raise ValueError("missing definition keys at category {}".format(i))

        cat = __create_attr_cat__(cat_def["cat_label"], cat_def["df_slct_f"], cat_def["s_slct_f"])
        cats4attr.append(cat)

        i += 1

    return cats4attr


def __templ_df_slct_f_by_vals__(attr: str, val: Any) -> pd.Series:
    return lambda df: df[attr] == val


def __templ_s_slct_f_by_vals__(attr: str, val: Any) -> bool:
    return lambda s: s[attr] == val


def categorize_attr_by_vals(attr: str, attr_vals: list[Any]) -> list[AttrCat]:
    cats4attr = list()

    for val in attr_vals:
        df_slct_f = __templ_df_slct_f_by_vals__(attr, val)
        s_slct_f = __templ_s_slct_f_by_vals__(attr, val)

        cat = __create_attr_cat__(val, df_slct_f, s_slct_f)
        cats4attr.append(cat)
    return cats4attr


def categorize_attrs_by_vals_from_df(df: pd.DataFrame, attrs: list[str] = None, attrs_vals: dict[str, list[Any]] = None,
                                     cats4attrs: dict[str, list[AttrCat]] = None) -> dict[str, list[AttrCat]]:
    if attrs is None:
        attrs = df.columns

    attrs_vals_provided = False
    if attrs_vals is not None:
        attrs_vals_provided = True

    if cats4attrs is None:
        cats4attrs = dict()

    for attr in attrs:
        if attr not in cats4attrs:
            attr_vals = df[attr].unique() if not attrs_vals_provided or attr not in attr_vals \
                else attr_vals[attr]

            cats4attrs[attr] = categorize_attr_by_vals(attr, attr_vals)

    return cats4attrs


def apply_cats_to_df(df: pd.DataFrame, cats4attr: list[AttrCat], deep_copy: bool = False) -> dict[Any, pd.DataFrame]:
    cat_dfs = dict()
    for cat in cats4attr:
        cat_dfs[cat.cat_label] = df[cat.slct_f(df)].copy(deep=deep_copy)
    return cat_dfs


def __label_shannon_entropy__(positive_cases: int, total_cases: int) -> float:
    if positive_cases == 0:
        return 0
    relative_probability = positive_cases / total_cases
    return - relative_probability * math.log2(relative_probability)


def shannon_entropy(X: pd.Series) -> float:
    label_count = X.value_counts()
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


def __attr_val_gain__(Sv: pd.DataFrame, out_label: str, S_count: int) -> float:
    Sv_count = Sv.shape[0]
    h_Sv = shannon_entropy(Sv[out_label])

    Sv_gain = (Sv_count / S_count) * h_Sv
    return Sv_gain


def __attr_gain__(S: pd.DataFrame, cats4attr: list[AttrCat], out_label: str, S_count: int) -> float:
    attr_gain = 0
    for cat in cats4attr:
        Sv = S[cat.slct_f(S)]
        Sv_gain = __attr_val_gain__(Sv, out_label, S_count)
        attr_gain += Sv_gain
    return attr_gain


def gain(S: pd.DataFrame, out_label: str, attr: str = None, attr_vals: list[str] = None,
         cats4attr: list[AttrCat] = None) -> float:
    if cats4attr is None and attr is None:
        raise ValueError("must provide either cats4attr or attr. Both cannot be None")

    S_count = S.shape[0]
    h_S = shannon_entropy(S[out_label])

    if cats4attr is None:
        if attr_vals is None:
            attr_vals = S[attr].unique()

        cats4attr = categorize_attr_by_vals(attr, attr_vals)

    attr_gain = __attr_gain__(S, cats4attr, out_label, S_count)
    return h_S - attr_gain

@dataclass
class PrePruning:
    max_depth : int = None


def pre_pruning_from_dict(pre_pruning_dict : dict[str, Any]) -> PrePruning:
    pre_pruning = PrePruning()

    def set_max_depth(pre_pruning : PrePruning, max_depth : int):
        pre_pruning.max_depth = max_depth

    pre_pruning_criteria = { 
        "max_depth" : set_max_depth
    }

    for criteria in pre_pruning_criteria.keys():
        if criteria in pre_pruning_dict:
            pre_pruning_criteria[criteria](pre_pruning, pre_pruning_dict[criteria])

    return pre_pruning

@dataclass
class DecisionTreeProperties:
    depth           : int
    attr_node_count : int
    val_node_count  : int
    leaf_node_count : int

    def node_count(self) -> int:
        return self.attr_node_count + self.val_node_count + self.leaf_node_count