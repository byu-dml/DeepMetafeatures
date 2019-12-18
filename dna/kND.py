# adapted from https://github.com/automl/auto-sklearn/blob/master/autosklearn/metalearning/metalearning/kNearestDatasets/kND.py

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
import sklearn.utils


class MinMaxScaler:

    def __init__(self):
        self._mins = None
        self._maxs = None

    def fit(self, X):
        assert isinstance(X, pd.DataFrame), type(X)
        self._mins = X.min(axis=0)
        self._maxs = X.max(axis=0)

    def transform(self, X):
        return (X - self._mins) / (self._maxs - self._mins)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KNearestDatasets(object):

    def __init__(self, metric='l1', metric_params=None):
        self.metric = metric
        if callable(self.metric):
            self._metric = self.metric
            self._p = 0
        elif self.metric.lower() == 'l1':
            self._metric = 'minkowski'
            self._p = 1
        elif self.metric.lower() == 'l2':
            self._metric = 'minkowski'
            self._p = 2
        else:
            raise ValueError(self.metric)
        self.metric_params = metric_params if metric_params is not None else {}

        self.metafeatures = None
        self._scaled_metafeatures = None
        self.runs = None
        self._nearest_neighbors = None
        self._min_max_scaler = MinMaxScaler()

    def fit(self, metafeatures, runs):
        """
        Fit the Nearest Neighbors model.

        Parameters
        ----------
        metafeatures : pandas.DataFrame
            A pandas dataframe: each row represents a dataset and each column a metafeature.
        runs : pandas.DataFrame
            A pandas dataframe: each row represents a pipeline run and each column a dataset.
        """
        assert isinstance(metafeatures, pd.DataFrame)
        assert metafeatures.values.dtype in (np.float32, np.float64)
        assert np.isfinite(metafeatures.values).all()
        assert isinstance(runs, pd.DataFrame)
        assert np.isfinite(runs).any(axis=1).all()
        assert runs.shape[1] == metafeatures.shape[0], 'number of datasets mismatch: {} {}'.format(
            runs.shape[1], metafeatures.shape[0]
        )

        self.metafeatures = metafeatures
        self._scaled_metafeatures = self._min_max_scaler.fit_transform(self.metafeatures)
        self.runs = runs
        self.num_datasets = runs.shape[1]

        self._nearest_neighbors = NearestNeighbors(
            n_neighbors=self.num_datasets, radius=None, algorithm='brute', leaf_size=30, metric=self._metric,
            p=self._p, metric_params=self.metric_params
        )
        self._nearest_neighbors.fit(self._scaled_metafeatures)

    def kNearestDatasets(self, x, k=1, return_distance=False):
        """
        Return the k most similar datasets with respect to self.metric

        Parameters
        ----------
        x : pandas.Series
            Metafeatures for one dataset
        k : int
            Number of k nearest datasets which are returned. If k == -1, return all dataset sorted by similarity.
        return_distance : bool, optional. Defaults to False
            If true, distances to the new dataset will be returned.

        Returns
        -------
        list
            Names of the most similar datasets, sorted by similarity
        list
            Sorted distances. Only returned if return_distances is set to True.
        """
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        elif k == -1:
            k = self.num_datasets

        x = pd.DataFrame(x.values.reshape(1,-1))
        x = self._min_max_scaler.transform(x)
        distances, neighbor_indices = self._nearest_neighbors.kneighbors(x, n_neighbors=k, return_distance=True)

        assert k == neighbor_indices.shape[1]

        # Neighbor indices is 2d, each row are the indices for one dataset in x.
        rval = [self.metafeatures.index[i] for i in neighbor_indices[0]]

        if return_distance is False:
            return rval
        else:
            return rval, distances[0]

    def kBestSuggestions(self, x, k=1, exclude_double_configurations=True):
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')

        nearest_datasets, distances = self.kNearestDatasets(x, -1, return_distance=True)
        kbest = []

        added_configurations = set()
        # get the top 25 best pipelines from each dataset
        for dataset_name, distance in zip(nearest_datasets, distances):
            best_configuration = self.runs[dataset_name].idxmax()

            if best_configuration is None:
                continue

            if exclude_double_configurations:
                # make sure the best pipelines we're returning have actual values and are not already part of our top 25
                if best_configuration not in added_configurations:
                    added_configurations.add(best_configuration)
                    kbest.append(best_configuration)
            else:
                kbest.append(best_configuration)

            if k != -1 and len(kbest) >= k:
                break

        if k == -1:
            k = len(kbest)

        return kbest[:k]

    def knn_regression(self, x):
        """
        Regress performance of a pipeline from the training set for a given set of metafeatures x using KNN regression.

        Parameters
        ----------
        x: pandas.Series
            The metafeatures of some dataset
        """
        assert type(x) == pd.Series

        nearest_datasets, distances = self.kNearestDatasets(x, k=-1, return_distance=True)
        if 0. in distances:
            distances = [int(dist == 0.) for dist in distances]
        distances = pd.Series(distances, nearest_datasets).reindex(self.runs.columns)

        regressed_values = {}
        for pipeline_id, dataset_scores in self.runs.iterrows():
            denom = distances[~dataset_scores.isnull()].sum()
            if denom > 0.:
                num = (dataset_scores * distances).sum()
                regressed_values[pipeline_id] = num / denom

        return pd.Series(regressed_values)
