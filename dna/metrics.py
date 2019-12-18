import collections
import typing
import warnings

import numpy as np
import scipy.stats
from sklearn.metrics import accuracy_score, mean_squared_error

from dna.models.model_utils import pearson_loss


def accuracy(y_hat, y):
    return accuracy_score(y, y_hat)


def rmse(y_hat, y):
    return mean_squared_error(y, y_hat)**.5


def pearson_loss_metric(y_hat, y):
    assert type(y_hat) == list
    assert type(y) == list
    y_hat = np.array(y_hat)
    y = np.array(y)
    return pearson_loss(y_hat, y).item()


def pearson_correlation(y_hat, y):
    """
    Calculates Pearson's R^2 coefficient. Returns a tuple containing the correlation coefficient and the p value for
    the test that the correlation coefficient is different than 0.
    """
    with warnings.catch_warnings(record=True) as w:
        correlation, pvalue = scipy.stats.pearsonr(y_hat, y)

        if np.isnan(correlation):
            print('WARNING: PEARSON CORRELATION IS NAN')
            correlation = 0.0

        return correlation, pvalue


def spearman_correlation(x: typing.Sequence, y: typing.Sequence):
    with warnings.catch_warnings(record=True) as w:
        spearman = scipy.stats.spearmanr(x, y)
        correlation = spearman.correlation
        pvalue = spearman.pvalue

        if np.isnan(correlation):
            print('WARNING: SPEARMAN CORRELATION IS NAN')
            correlation = 0.0

        return correlation, pvalue


def _get_rank_order_relevance(relevance: typing.Sequence, rank: typing.Sequence):
    if rank is None:
        return np.array(relevance)
    else:
        assert len(rank) <= len(relevance)
        rank_order = np.argsort(rank)
        return np.array(relevance)[rank_order]


def _get_values_at_k(
    values: np.array, k: typing.Optional[typing.Union[typing.Iterable, int]]
) -> typing.Union[typing.Iterable, int]:
    if k is None:
        return values.tolist()
    elif isinstance(k, collections.Iterable):
        return values[np.array(k) - 1].tolist()
    else:
        return values[k - 1]


def n_correct_at_k(relevance: typing.Sequence, rank: typing.Sequence = None, k: typing.Union[int, typing.Iterable] = None):
    """Counts the number of top-k ranked items that have the top-k relevancy. Optimized to compute n_correct_at_k
    for all k. In the case of relevancy ties, all tied relevancies are included in top-k.
    """
    sorted_rel = sorted(relevance)[::-1]
    rank_order_rel = _get_rank_order_relevance(relevance, rank)

    correct = np.arange(1, len(rank_order_rel) + 1)
    for i, r in enumerate(rank_order_rel):
        j = i
        while r < sorted_rel[j]:
            correct[j] -= 1
            j += 1

    return _get_values_at_k(correct, k)


def regret_at_k(relevance: typing.Sequence, rank: typing.Sequence = None, k: int = None):
    """Computes the difference between the highest relevance item with the ranked highest relevance item. Optimized
    to compute for all k.
    """
    best = max(relevance)
    rank_order_rel = _get_rank_order_relevance(relevance, rank)
    regret = np.minimum.accumulate(best - rank_order_rel)
    return _get_values_at_k(regret, k)


def _dcg(
    relevance: typing.Sequence, rank: typing.Sequence = None, gains_f: str = 'exponential'
):
    rank_order_rel = _get_rank_order_relevance(relevance, rank)

    if gains_f == 'exponential':
        gains = 2 ** rank_order_rel - 1
    elif gains_f == 'linear':
        gains = rank_order_rel
    else:
        raise ValueError('Invalid gains_f: {}'.format(gains_f))

    # discount = log2(i + 1), with i starting at 1
    discounts = np.log2(np.arange(2, len(gains) + 2))
    discounted_gains = gains / discounts
    return np.cumsum(discounted_gains)

def dcg_at_k(
    relevance: typing.Sequence, rank: typing.Sequence = None, k: typing.Union[int, typing.Iterable] = None,
    gains_f: str = 'exponential'
):
    """Discounted cumulative gain (DCG) at rank k. Optimized to compute dcg@k for all k.

    Parameters
    ----------
    relevance: Sequence
        True relevance labels
    rank: Sequence
        Predicted rank for actual_relevance. If not provided, actual_relevance is assumed to be in rank order.
    k: Union[int, Iterable[int]]
        Rank position. If k is an int, dcg will be computed at k. If k is an iterable, dcg will be computed at all
        values in k. If k is None, all possible values of k will be used.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    idcg: bool
        the ideal ordering of the dcg, used for calculating the ndcg

    Returns
    -------
    DCG@k: float
    """
    dcg = _dcg(relevance, rank, gains_f)
    return _get_values_at_k(dcg, k)


def ndcg_at_k(
    relevance: typing.Sequence, rank: typing.Sequence = None, k: typing.Union[int, typing.Iterable] = None,
    gains_f: str = 'exponential'
):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    relevance: Sequence
        True relevance labels
    rank: Sequence
        Predicted rank for relevance. If not provided, relevance is assumed to be in rank order.
    k: int
        Rank position.
    gains: str
        Whether gains should be "exponential" (default) or "linear".

    Returns
    -------
    NDCG@k: float
    """
    dcg = _dcg(relevance, rank, gains_f=gains_f)
    idcg = _dcg(np.sort(relevance)[::-1], gains_f=gains_f)
    _ndcg_at_k = np.array(dcg) / np.array(idcg)
    return _get_values_at_k(_ndcg_at_k, k)
