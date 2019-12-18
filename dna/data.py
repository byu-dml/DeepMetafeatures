import json
import os
import random
import tarfile
import typing

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import dtype


def group_json_objects(json_objects, group_key):
    """
    Groups JSON data by group_key.

    Parameters:
    -----------
    json_objects: List[Dict], JSON compatible list of objects.
    group_key: str, json_objects is grouped by group_key. group_key must be a
        key into each object in json_objects and the corresponding value must
        be hashable. group_key can be a '.' delimited string to access deeply
        nested fields.

    Returns:
    --------
    A dict with key being a group and the value is a list of indices into
    json_objects.
    """
    grouped_objects = {}
    for i, obj in enumerate(json_objects):
        group = obj
        for key_part in group_key.split('.'):
            group = group[key_part]
        if group not in grouped_objects:
            grouped_objects[group] = []
        grouped_objects[group].append(i)
    return grouped_objects


def split_data_by_group(data: typing.List[typing.Dict], group_by_key: str, test_size: typing.Union[int, float], seed: int):
    grouped_data_indices = group_json_objects(data, group_by_key)
    groups = list(grouped_data_indices.keys())

    if 0 < test_size < 1:
        test_size = int(round(test_size * len(groups)))
    if test_size <= 0 or len(groups) <= test_size:
        raise ValueError('invalid test size: {}'.format(test_size))

    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(groups)

    train_groups = groups[test_size:]
    assert len(train_groups) == len(groups) - test_size
    test_groups = groups[:test_size]
    assert len(test_groups) == test_size

    train_data = []
    for group in train_groups:
        for i in grouped_data_indices[group]:
            train_data.append(data[i])

    test_data = []
    for group in test_groups:
        for i in grouped_data_indices[group]:
            test_data.append(data[i])

    return train_data, test_data


def _extract_tarfile(path):
    assert tarfile.is_tarfile(path)

    dirname = os.path.dirname(path)
    with tarfile.open(path, 'r:*') as tar:
        members = tar.getmembers()
        if len(members) != 1:
            raise ValueError('Expected tar file with 1 member, but got {}'.format(len(members)))
        tar.extractall(os.path.dirname(path))
        extracted_path = os.path.join(dirname, tar.getmembers()[0].name)

    return extracted_path


def get_data(path):
    if tarfile.is_tarfile(path):
        path = _extract_tarfile(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class DropMissingValues:

    def __init__(self, values_to_drop=[]):
        self.values_to_drop = values_to_drop

    def fit(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        for key, is_missing in pd.DataFrame(data).isna().any().iteritems():
            if is_missing:
                self.values_to_drop.append(key)

    def predict(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        for instance in data:
            for key in self.values_to_drop:
                instance.pop(key, None)
        return data


class StandardScaler:
    """
    Transforms data by subtracting the mean and scaling by the standard
    deviation. Drops columns that have 0 standard deviation. Clips values to
    numpy resolution, min, and max.
    """

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        values_map = {}
        for instance in data:
            for key, value in instance.items():
                if key not in values_map:
                    values_map[key] = []
                values_map[key].append(value)

        self.means = {}
        self.stds = {}
        for key, values in values_map.items():
            self.means[key] = np.mean(values)
            self.stds[key] = np.std(values, ddof=1)

    def predict(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        if self.means is None or self.stds is None:
            raise Exception('StandardScaler not fit')

        transformed_data = []
        for instance in data:
            transformed_instance = {}
            for key, value in instance.items():
                if self.stds[key] != 0:  # drop columns with 0 std dev
                    transformed_instance[key] = (value - self.means[key]) / self.stds[key]

            transformed_data.append(transformed_instance)

        return transformed_data


def encode_dag(dag):
    """
    Converts a directed acyclic graph DAG) to a string. If two DAGs have the same encoding string, then they are equal.
    However, two isomorphic DAGs may have different encoding strings.

    Parameters
    ----------
    dag:
        A representation of a dag. Each element in the outer list represents a vertex. Each inner list or vertex
        contains a reference to the outer list, representing edges.
    """
    return ''.join(''.join(str(edge) for edge in vertex) for vertex in dag)


def filter_metafeatures(metafeatures: dict, metafeature_subset: str):
    landmarker_key_part1 = 'ErrRate'
    landmarker_key_part2 = 'Kappa'

    metafeature_keys = list(metafeatures.keys())

    if metafeature_subset == 'landmarkers':
        for metafeature_key in metafeature_keys:
            if landmarker_key_part1 not in metafeature_key and landmarker_key_part2 not in metafeature_key:
                metafeatures.pop(metafeature_key)
    elif metafeature_subset == 'non-landmarkers':
        for metafeature_key in metafeature_keys:
            if landmarker_key_part1 in metafeature_key or landmarker_key_part2 in metafeature_key:
                metafeatures.pop(metafeature_key)

    return metafeatures


def preprocess_data(train_data, test_data, metafeature_subset: str):
    for instance in train_data:
        instance['pipeline_id'] = instance['pipeline']['id']

    for instance in test_data:
        instance['pipeline_id'] = instance['pipeline']['id']

    train_metafeatures = []
    for instance in train_data:
        metafeatures = filter_metafeatures(instance['metafeatures'], metafeature_subset)
        train_metafeatures.append(metafeatures)
        for step in instance['pipeline']['steps']:
            step['name'] = step['name'].replace('.', '_')

    test_metafeatures = []
    for instance in test_data:
        metafeatures = filter_metafeatures(instance['metafeatures'], metafeature_subset)
        test_metafeatures.append(metafeatures)
        for step in instance['pipeline']['steps']:
            step['name'] = step['name'].replace('.', '_')

    # drop metafeature if missing for any instance
    dropper = DropMissingValues()
    dropper.fit(train_metafeatures)
    train_metafeatures = dropper.predict(train_metafeatures)
    test_metafeatures = dropper.predict(test_metafeatures)

    # scale data to unit mean and unit standard deviation
    scaler = StandardScaler()
    scaler.fit(train_metafeatures)
    train_metafeatures = scaler.predict(train_metafeatures)
    test_metafeatures = scaler.predict(test_metafeatures)

    # Primitives to remove from the pipelines
    # These primitives are important for the actual execution of the pipeline but they say nothing about performance
    irrelevant_primitives: set = {
        'd3m_primitives_data_transformation_dataset_to_dataframe_Common',
        'd3m_primitives_data_transformation_column_parser_DataFrameCommon',
        'd3m_primitives_data_preprocessing_random_sampling_imputer_BYU',
        'd3m_primitives_data_transformation_extract_columns_by_semantic_types_DataFrameCommon',
        'd3m_primitives_data_transformation_construct_predictions_DataFrameCommon',
        'd3m_primitives_data_transformation_rename_duplicate_name_DataFrameCommon'
    }

    # convert from dict to list
    for instance, mf_instance in zip(train_data, train_metafeatures):
        _preprocess_instance(instance, mf_instance, irrelevant_primitives)

    for instance, mf_instance in zip(test_data, test_metafeatures):
        _preprocess_instance(instance, mf_instance, irrelevant_primitives)

    return train_data, test_data


def _preprocess_instance(instance: dict, mf_instance, irrelevant_primitives: set):
    instance['metafeatures'] = [value for key, value in sorted(mf_instance.items())]

    _remove_irrelevant_primitives(instance, irrelevant_primitives)

    pipeline_dag = [step['inputs'] for step in instance['pipeline']['steps']]
    instance['pipeline_structure'] = encode_dag(pipeline_dag)
    instance['pipeline_len'] = len(pipeline_dag)


def _remove_irrelevant_primitives(instance: dict, irrelevant_primitives: set):
    steps: list = instance['pipeline']['steps']
    steps_to_remove: list = []
    for i, step in enumerate(steps):
        # If the primitive of this step is irrelevant to a meta model
        if step['name'] in irrelevant_primitives:
            # Indicate that the step should be removed
            steps_to_remove.append(i)

            # Execute the reconstruction process for the removal of this step
            # Do this by connecting its inputs to each step that take the removed step as an input
            for j in range(len(steps)):
                # Get the inputs of the current step
                inputs: list = steps[j]['inputs']

                # If the current step takes the removed step as input
                if i in inputs:
                    # Remove the current_steps's input which corresponds to the removed step
                    inputs.remove(i)

                    # Add the inputs of the removed step to the current step
                    for idx in step['inputs']:
                        if idx not in inputs:
                            inputs.append(idx)

    # Remove the irrelevant primitives
    steps_to_remove = np.array(steps_to_remove)
    for idx in steps_to_remove:
        del steps[idx]

        # For each remaining step
        for step in steps:
            # For each input of the current step
            inputs = step['inputs']
            for i in range(len(inputs)):
                # If the input is not null and is greater than the index of the removed step
                if type(inputs[i]) is int and inputs[i] > idx:
                    # Subtract 1 from the input since all the steps beyond the removed step were moved back in the list
                    inputs[i] -= 1
        # Subtract 1 from the indices of the rest of the steps to remove since those too were moved back in the list
        steps_to_remove -= 1

    # Standardize the inputs which were made incorrect
    # They are made incorrect in the primitive removal and pipeline reconstruction process above
    for step in steps:
        inputs = step['inputs']

        # Make sure the null inputs are by themselves to avoid both redundancy and inputs sizes more than 2
        if 'inputs.0' in inputs and len(inputs) > 1:
            inputs.remove('inputs.0')

        # Make sure there are no duplicate inputs and that they are ordered for consistency
        step['inputs'] = sorted(set(inputs))

        # Ensure that there are never more than 2 inputs into a step
        assert len(step['inputs']) <= 2, 'The inputs of this step exceed 2'


class Dataset(torch.utils.data.Dataset):
    """
    A subclass of torch.utils.data.Dataset for handling simple JSON structed
    data.

    Parameters:
    -----------
    data: List[Dict], JSON structed data.
    features_key: str, the key into each element of data whose value is a list
        of features used for input to a PyTorch network.
    target_key: str, the key into each element of data whose value is the
        target used for a PyTorch network.
    device": str, the device onto which the data will be loaded
    """

    def __init__(
        self, data: typing.List[typing.Dict], features_key: str,
        target_key: str, device: str
    ):
        self.data = data
        self.features_key = features_key
        self.target_key = target_key
        self.device = device

    def __getitem__(self, item: int):
        x = torch.tensor(self.data[item][self.features_key], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data[item][self.target_key], device=self.device)
        return x, y

    def __len__(self):
        return len(self.data)


class RandomSampler(torch.utils.data.Sampler):
    """
    Samples indices uniformly without replacement.

    Parameters
    ----------
    n: int
        the number of indices to sample
    seed: int
        used to reproduce randomization
    """

    def __init__(self, n, seed):
        self.n = n
        self._indices = list(range(n))
        self._random = random.Random()
        self._random.seed(seed)

    def __iter__(self):
        self._random.shuffle(self._indices)
        return iter(self._indices)

    def __len__(self):
        return self.n


class GroupDataLoader(object):
    """
    Batches a dataset for PyTorch Neural Network training. Partitions the
    dataset so that batches belong to the same group.

    Parameters:
    -----------
    data: List[Dict], JSON compatible list of objects representing a dataset.
        dataset_class must know how to parse the data given dataset_params.
    group_key: str, pipeline run data is grouped by group_key and each
        batch of data comes from only one group. group_key must be a key into
        each element of the pipeline run data. the value of group_key must be
        hashable.
    dataset_class: Type[torch.utils.data.Dataset], the class used to make
        dataset instances after the dataset is partitioned.
    dataset_params: dict, extra parameters needed to instantiate dataset_class
    batch_size: int, the number of data points in each batch
    drop_last: bool, default False. whether to drop the last incomplete batch.
    shuffle: bool, default True. whether to randomize the batches.
    """

    def __init__(
        self, data: typing.List[typing.Dict], group_key: str,
        dataset_class: typing.Type[torch.utils.data.Dataset], dataset_params: dict,
        batch_size: int, drop_last: bool, shuffle: bool, seed: int
    ):
        self.data = data
        self.group_key = group_key
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self._random = random.Random()
        self._random.seed(seed)
        self.old_indices = []

        self._init_dataloaders()
        self._init_group_metadataloader()

    def _init_dataloaders(self):
        """
        Groups self.data based on group_key. Creates a
        torch.utils.data.DataLoader for each group, using self.dataset_class.
        """
        # group the data
        grouped_data = group_json_objects(self.data, self.group_key)

        # create dataloaders
        self._group_dataloaders = {}
        for group, group_indices in grouped_data.items():
            self.old_indices += group_indices
            group_data = [self.data[i] for i in group_indices]
            group_dataset = self.dataset_class(group_data, **self.dataset_params)
            new_dataloader = self._get_data_loader(
                group_dataset
            )
            self._group_dataloaders[group] = new_dataloader

    def _get_data_loader(self, data):
        if self.shuffle:
            sampler = RandomSampler(len(data), self._randint())
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset=data, sampler=sampler, batch_size=self.batch_size, drop_last=self.drop_last
        )
        return dataloader

    def _randint(self):
        return self._random.randint(0,2**32-1)

    def _init_group_metadataloader(self):
        """
        Creates a dataloader which randomizes the batches over the groups. This
        allows the order of the batches to be independent of the groups.
        """
        self._group_batches = []
        for group, group_dataloader in self._group_dataloaders.items():
            self._group_batches += [group] * len(group_dataloader)
        if self.shuffle:
            self._random.shuffle(self._group_batches)

    def get_group_ordering(self):
        """
        Returns the indices needed to invert the ordering on the input data generated by the grouping mechanism. This
        method does not work if shuffle or drop last has been set to true.
        """
        if self.shuffle or self.drop_last:
            raise NotImplementedError('cannot ungroup data when shuffle is true or drop_last is true')
        return np.argsort(np.array(self.old_indices))

    def __iter__(self):
        return iter(self._iter())

    def _iter(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self._group_batches)


class DAGDataset(Dataset):

    def __init__(self, data: list, features_key: str, target_key: str, device: str, input_type: dtype, mask_key: str):
        super().__init__(data, features_key, target_key, device)

        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.input_type = input_type
        self.mask_key = mask_key

    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        item = self.data[index]
        encoded_pipeline = torch.tensor(
            item[self.pipeline_key][self.steps_key], dtype=self.input_type, device=self.device
        )

        # If we are masking the pipeline according to its DAG structure
        if self.mask_key in item[self.pipeline_key]:
            pipeline_mask = torch.tensor(
                item[self.pipeline_key][self.mask_key], dtype=torch.float32, device=self.device
            )
            return pipeline_mask, encoded_pipeline, x, y

        return encoded_pipeline, x, y


class DAGDataLoader(GroupDataLoader):

    def __init__(
        self, data: list, group_key: str, dataset_params: dict, batch_size: int, drop_last: bool, shuffle: bool,
        seed: int, encode_pipeline: callable, yield_batch: callable, pipelines_encoded: callable
    ):
        super().__init__(data, group_key, DAGDataset, dataset_params, batch_size, drop_last, shuffle, seed)

        self.yield_batch = yield_batch

        if not pipelines_encoded(data):
            self._encode_pipelines(
                data, encode_pipeline
            )

    @staticmethod
    def _encode_pipelines(
            data, encode_pipeline
    ):
        for instance in data:
            encode_pipeline(instance)

    def _iter(self):
        group_dataloader_iters = {}
        for group in self._group_batches:
            if group not in group_dataloader_iters:
                group_dataloader_iters[group] = iter(self._group_dataloaders[group])

            # Get a batch of encoded pipelines, metafeatures, targets, and other optional inputs
            yield self.yield_batch(group_dataloader_iters, group)
        raise StopIteration()


class MetaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_ids, seed):
        self.seed = seed

        root_dirs = [
            'datasets/LL0',
            'datasets/seed_datasets_current'
        ]

        # Validate the data set root directories
        for root_dir in root_dirs:
            if not os.path.isdir(root_dir):
                raise NotADirectoryError(
                    'The dataset root directory, {}, was not found. You need to extract the datasets'.format(root_dir)
                )

        self.dataset_dirs = []
        for root_dir in root_dirs:
            for elem in os.listdir(root_dir):
                elem = os.path.join(root_dir, elem)
                if os.path.isdir(elem):
                    self.dataset_dirs.append(elem)

        self.dataset_id_key = 'dataset_id'
        self.dataset_key = 'dataset'

        self.datasets = []
        for dataset_id in dataset_ids:
            if dataset_id.endswith('_dataset'):
                dataset_name = dataset_id[:-len('_dataset')]
            else:
                dataset_name = dataset_id
            dataset = self._get_dataset(dataset_name, dataset_id)

            if dataset is not None:
                item = {
                    self.dataset_id_key: dataset_id,
                    self.dataset_key: dataset
                }
                self.datasets.append(item)

    def _get_dataset(self, dataset_name, dataset_id):
        data = None
        for dataset_dir in self.dataset_dirs:
            if dataset_name in dataset_dir:
                df_path: str = os.path.join(dataset_dir, dataset_id, 'tables', 'learningData.csv')
                df = pd.read_csv(df_path, na_values='?')
                with open(os.path.join(dataset_dir, dataset_name+'_dataset', 'datasetDoc.json'), 'r') as fp:
                    metadata = json.load(fp)

                def print_exception(msg: str):
                    print('###########################################################################################')
                    print(msg)
                    import traceback
                    traceback.print_exc()
                    print('###########################################################################################')

                try:
                    data = self._preprocess_data(df, metadata)
                except KeyError:
                    print_exception('Data set at {} does not contain a d3m index column'.format(df_path))
                    return None
                except ValueError:
                    print_exception('Dataset at {} could not sample due to the following exception:'.format(df_path))
                    return None

                break

        assert data is not None, 'dataset_id: {}'.format(dataset_id)
        return data

    def sample_column(self, col):
        na_vals = col.isna()
        n_nan = na_vals.sum()
        valid_vals = col.dropna()
        if n_nan > 0:
            col.loc[na_vals] = valid_vals.sample(n=n_nan, replace=True, random_state=self.seed).to_list()
        return col

    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return data

    def _preprocess_data(self, data, metadata):
        if data.isna().sum().sum():
            data.apply(self.sample_column, axis=0, result_type='broadcast')

        # TODO: Find more sophisticated way to deal with string/int/categorical columns
        cat_cols = []
        targets = None
        for col in metadata['dataResources'][0]['columns']:
            col_name = col['colName']

            if col_name == 'd3mIndex':
                data = data.drop(col_name, axis=1)
                continue

            n_unique_vals = len(data[col_name].unique())
            if n_unique_vals == 1:
                data = data.drop(col_name, axis=1)
            elif col['role'][0] == 'suggestedTarget':
                targets = data[col_name]
                data = data.drop(col_name, axis=1)
            elif col['colType'] in ['categorical', 'string']:
                if n_unique_vals > 25:
                    data = data.drop(col_name, axis=1)
                else:
                    cat_cols.append(col_name)

        # One-hot encode the categorical features and normalize the real features
        data = pd.get_dummies(data, columns=cat_cols)
        features = data.to_numpy()
        features = self._normalize_data(features)

        # Ordinal encode and normalize the targets
        assert targets is not None
        targets, _ = pd.factorize(targets)
        targets = np.expand_dims(targets, axis=-1)
        targets = self._normalize_data(targets)

        # Recombine the targets and features
        data = np.concatenate((features, targets), axis=-1)

        # Convert the dataset to a tensor and permute the dimensions such that the columns come first
        data = torch.from_numpy(data).to(torch.float)
        # TODO: add more info to the 3rd dim (ie cat vs num)
        data = data.unsqueeze(dim=-1).permute(1, 0, 2)  # [seq_len, batch_size, dim]
        return data

    def __getitem__(self, item):
        item = self.datasets[item]

        dataset = item[self.dataset_key]
        col_dim: int = 0
        col_len: int = dataset.shape[col_dim]
        row_dim: int = 1
        row_len = dataset.shape[row_dim]

        # Randomly sample across the columns
        col_indices = np.random.RandomState(seed=self.seed).permutation(col_len)
        dataset = dataset[col_indices]

        # Randomly sample across the rows
        row_indices = np.random.RandomState(seed=self.seed).permutation(row_len)
        dataset = dataset[:, row_indices, :]

        return item[self.dataset_id_key], dataset

    def __len__(self):
        return len(self.datasets)


class MetaDataLoader:

    def __init__(self, data, batch_size, get_dataset_dataloader, metafeatures_type, device, seed):
        dataset_id_key = 'dataset_id'
        grouped_by_dataset = group_json_objects(data, dataset_id_key)

        # Create the meta data loader which will load the data sets
        meta_dataset = MetaDataset(dataset_ids=grouped_by_dataset.keys(), seed=seed)
        self.meta_data_loader = torch.utils.data.DataLoader(
            dataset=meta_dataset, batch_size=1, drop_last=False, sampler=RandomSampler(len(meta_dataset), seed)
        )

        # Create a data loader for each data set in the meta data set
        # Each data loader will load pipelines, targets, and traditional metafeatures for its corresponding data set
        # The traditional metafeatures will be discarded and replaced if using ONLY deep metafeatures
        self.dataset_dataloaders = {}
        for dataset_id, dataset_indices in grouped_by_dataset.items():
            dataset_data = [data[i] for i in dataset_indices]
            self.dataset_dataloaders[dataset_id] = get_dataset_dataloader(
                dataset_data, batch_size, drop_last=False, shuffle=True
            )

        self.use_deep_metafeatures = metafeatures_type == 'deep' or metafeatures_type == 'both'
        self.device = device

    def __iter__(self):
        return iter(self._iter())

    def _iter(self):
        for dataset_id, dataset in self.meta_data_loader:
            # Since the dataset id and dataset are coming from a PyTorch data loader, they each are in a tuple
            dataset_id = dataset_id[0]
            dataset = dataset[0].to(self.device) if self.use_deep_metafeatures else None

            dataset_dataloader = self.dataset_dataloaders[dataset_id]
            yield dataset_id, dataset_dataloader, dataset
        raise StopIteration()

    def __len__(self):
        return len(self.meta_data_loader)
