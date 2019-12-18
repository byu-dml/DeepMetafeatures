import collections
import typing
import pandas as pd


def rank(values: typing.Sequence) -> typing.Sequence:
    return list(pd.Series(values).rank(ascending=False) - 1)


class RankProblemTargets:
    """Stores both the regression and rank targets necessary to score on the rank problem"""

    def __init__(self, regression_targets: dict, rank_targets: dict):
        self.regression_targets = regression_targets
        self.rank_targets = rank_targets


def flatten(d, parent_key='', sep='.'):
    # TODO: handle iterables
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def transpose_jagged_2darray(jagged_2darray: typing.Iterable[typing.Iterable]) -> typing.Dict[int, typing.List]:
    """Transposes a 2D jagged array into a dict mapping column index to a list of row values.

    For example:
        [
            [0, 1],
            [2, 3, 4],
            [5],
            [6, 7, 8, 9],
        ]
    becomes
        {
            0: [0, 2, 5, 6],
            1: [1, 3, 7],
            2: [4, 8],
            3: [9],
        }
    """
    transpose = {}
    for row in jagged_2darray:
        for i, value in enumerate(row):
            if i not in transpose:
                transpose[i] = []
            transpose[i].append(value)
    return transpose
