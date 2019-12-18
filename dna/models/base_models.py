import os

import numpy as np
import torch
from torch import dtype
import torch.nn as nn
from tqdm import tqdm

from dna import utils
from dna.models.model_utils import OptimizeCommand, pearson_loss, likelihood_loss
from dna.data import split_data_by_group, DAGDataLoader, MetaDataLoader
from dna.models.torch_modules.meta_model import MetaModel


class ModelBase:

    def __init__(self, *, seed):
        self.seed = seed
        self.fitted = False

    def fit(self, data, *, verbose=False):
        raise NotImplementedError()


class RegressionModelBase(ModelBase):

    def fit(self, data, *, verbose=False):
        raise NotImplementedError()

    def predict_regression(self, data, *, batch_size, verbose):
        raise NotImplementedError()


class RankModelBase(ModelBase):

    def fit(self, data, *, verbose=False):
        raise NotImplementedError()

    def predict_rank(self, data, *, batch_size, verbose):
        raise NotImplementedError()


class PyTorchModelBase:

    def __init__(
        self, *, device, seed, loss_function_name: str, loss_function_args: dict, metafeatures_type: str,
        metafeature_model_config
    ):
        self.device = device
        self.seed = seed
        self._validation_split_seed = seed + 1
        self._loss_function_name = loss_function_name
        self._loss_function_args = loss_function_args
        self.metafeatures_type = metafeatures_type

        if metafeatures_type == 'both' or metafeatures_type == 'deep':
            self.metafeature_model_config = metafeature_model_config
        elif metafeatures_type == 'traditional':
            self.metafeature_model_config = None

        self._loss_function = None
        self._optimizer = None
        self._model = None
        self._data_loader_seed = None
        self.fitted = None

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, validation_ratio: float, patience: int, *,
        output_dir=None, verbose=False
    ):
        # Split the validation and train data
        train_data, validation_data = self._get_validation_split(
            train_data, validation_ratio, self._validation_split_seed
        )

        self._model = self._get_model(train_data)
        self._loss_function = self._get_loss_function()
        train_data_loader = self._get_data_loader(train_data, batch_size)
        self._optimizer = self._get_optimizer(learning_rate)

        # Get the save paths for the meta model and the metafeature model
        model_save_path = None
        if output_dir is not None:
            model_save_path = os.path.join(output_dir, 'model.pt')

        # Get the validation data loader
        validation_data_loader = None
        if validation_data is not None:
            validation_data_loader = self._get_data_loader(validation_data, batch_size)

        min_loss_score = np.inf
        min_loss_epoch = -1

        if patience < 1:
            patience = np.inf

        save_model = None
        for e in range(n_epochs):
            save_model = False

            if verbose:
                print('epoch {}'.format(e))

            self._train_epoch(
                train_data_loader, self._model, self._loss_function, self._optimizer, verbose=verbose
            )

            train_predictions, train_targets = self._predict_epoch(
                train_data_loader, self._model, verbose=verbose
            )

            if verbose:
                mean_train_loss, std_train_loss = self._get_loss_by_dataset(train_predictions, train_targets)
                self._print_loss(mean_train_loss, std_train_loss, 'Train')

            validation_predictions, validation_targets = self._predict_epoch(
                validation_data_loader, self._model, verbose=verbose
            )
            mean_val_loss, std_val_loss = self._get_loss_by_dataset(validation_predictions, validation_targets)

            if verbose:
                self._print_loss(mean_val_loss, std_val_loss, 'Validation')

            if mean_val_loss < min_loss_score:
                min_loss_score = mean_val_loss
                min_loss_epoch = e
                save_model = True

            if save_model and model_save_path is not None:
                torch.save(self._model.state_dict(), model_save_path)

            if e - min_loss_epoch >= patience:
                break

        if not save_model and model_save_path is not None:  # model not saved during final epoch
            self._model.load_state_dict(torch.load(model_save_path))

        self.fitted = True

    @staticmethod
    def _print_loss(mean_loss: float, std_loss: float, loss_name: str):
        print('Mean {} Loss: {}, STD {} Loss: {}'.format(loss_name, mean_loss, loss_name, std_loss))

    def _get_loss_by_dataset(self, predictions: dict, targets: dict):
        losses = []
        for dataset_id in predictions:
            y = targets[dataset_id]
            y_hat = predictions[dataset_id]
            loss = self._loss_function(y_hat, y)
            losses.append(loss.item())
        mean = float(np.mean(losses))
        std = float(np.std(losses))
        return mean, std

    def _get_model(self, train_data):
        meta_model = self._get_meta_model(train_data)
        return MetaModel(meta_model, self.device, self.seed, self.metafeatures_type, self.metafeature_model_config)

    def _get_meta_model(self, train_data):
        raise NotImplementedError()

    def _get_dataset_dataloader(self, data, batch_size, drop_last, shuffle=True):
        raise NotImplementedError()

    def _get_loss_function(self):
        if self._loss_function_name == 'rmse':
            objective = torch.nn.MSELoss(reduction='mean')
            return lambda y_hat, y: torch.sqrt(objective(y_hat, y))
        elif self._loss_function_name == 'mse':
            return torch.nn.MSELoss(reduction='mean')
        elif self._loss_function_name == 'l1':
            return torch.nn.L1Loss(reduction='mean')
        elif self._loss_function_name == 'pearson_loss':
            return pearson_loss
        elif self._loss_function_name == 'likelihood_loss':
            return likelihood_loss
        else:
            raise ValueError('No valid loss function name provided. Got {}'.format(self._loss_function_name))

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size):
        return MetaDataLoader(
            data, batch_size, self._get_dataset_dataloader, self.metafeatures_type, self.device,
            self._data_loader_seed
        )

    @staticmethod
    def _process_dataset_predictions_and_targets(
            dataset_id, y_hat_by_dataset, y_by_dataset, y_hat, y, optimize_command: OptimizeCommand = None
    ):
        # If we are training, optimize on the predictions and targets of the current dataset
        if optimize_command is not None:
            optimize_command.run(y_hat, y)

        # Memory management
        y_hat_by_dataset[dataset_id] = y_hat.detach().cpu()
        y_by_dataset[dataset_id] = y.detach().cpu()

    def _get_dataset_batches(
        self, meta_data_loader, model, verbose, progress, optimize_command: OptimizeCommand = None
    ):
        y_hat_by_dataset = {}
        y_by_dataset = {}
        for dataset_id, dataset_data_loader, dataset in meta_data_loader:
            y_hat, y = model(dataset_data_loader, dataset, optimize_command=optimize_command)
            self._process_dataset_predictions_and_targets(
                dataset_id, y_hat_by_dataset, y_by_dataset, y_hat, y, optimize_command=optimize_command
            )

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

        return y_hat_by_dataset, y_by_dataset

    def _train_epoch(self, data_loader, model: nn.Module, loss_function, optimizer, *, verbose=True):
        model.train()

        progress = None
        if verbose:
            progress = tqdm(total=len(data_loader), position=0)

        optimizer_command = OptimizeCommand(loss_function, optimizer)

        # Get all the predictions and targets for each dataset but optimize on them rather than returning them
        self._get_dataset_batches(
            data_loader, model, verbose, progress, optimize_command=optimizer_command
        )

    def _predict_epoch(self, data_loader, model: nn.Module, *, verbose=True):
        model.eval()

        progress = None
        if verbose:
            progress = tqdm(total=len(data_loader), position=0)

        with torch.no_grad():
            # Get all the predictions and targets for each dataset
            y_hat_by_dataset, y_by_dataset = self._get_dataset_batches(data_loader, model, verbose, progress)

        return y_hat_by_dataset, y_by_dataset

    @staticmethod
    def _get_validation_split(train_data, validation_ratio, split_seed):
        if validation_ratio < 0 or 1 <= validation_ratio:
            raise ValueError('invalid validation ratio: {}'.format(validation_ratio))

        if validation_ratio == 0:
            return train_data, None

        return split_data_by_group(train_data, 'dataset_id', validation_ratio, split_seed)


class PyTorchRegressionRankModelBase(PyTorchModelBase, RegressionModelBase, RankModelBase):

    def __init__(
        self, seed, device, metafeatures_type, metafeature_model_config, loss_function_name, loss_function_args=None
    ):
        # different arguments means different function calls
        PyTorchModelBase.__init__(
            self, device=device, seed=seed, loss_function_name=loss_function_name,
            metafeature_model_config=metafeature_model_config, loss_function_args=loss_function_args,
            metafeatures_type=metafeatures_type
        )
        RegressionModelBase.__init__(self, seed=seed)

    def _get_meta_model(self, train_data):
        raise NotImplementedError()

    def _get_dataset_dataloader(self, data, batch_size, drop_last, shuffle=True):
        raise NotImplementedError()

    def predict_regression(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, batch_size)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)

        # Convert the predictions and targets to lists
        for dataset_id, dataset_predictions in predictions.items():
            dataset_targets = targets[dataset_id]
            predictions[dataset_id] = dataset_predictions.tolist()
            targets[dataset_id] = dataset_targets.tolist()

        return predictions, targets

    def predict_rank(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        regression_predictions, regression_targets = self.predict_regression(
            data, batch_size=batch_size, verbose=verbose
        )

        rank_predictions = {}
        rank_targets = {}
        for dataset_id, dataset_predictions in regression_predictions.items():
            dataset_targets = regression_targets[dataset_id]

            rank_predictions[dataset_id] = utils.rank(dataset_predictions)
            rank_targets[dataset_id] = utils.rank(dataset_targets)

        return rank_predictions, utils.RankProblemTargets(regression_targets, rank_targets)


class DAGRegressionRankModelBase(PyTorchRegressionRankModelBase):

    def __init__(self, meta_model_activation_name: str, batch_group_key: str, dataset_input_type: dtype, *super_args):

        super().__init__(*super_args)

        self.activation_name = meta_model_activation_name
        self._data_loader_seed = self.seed + 1
        self._model_seed = self.seed + 2
        self.num_primitives = None
        self.primitive_name_to_enc = None
        self.target_key = 'test_f1_macro'
        self.batch_group_key = batch_group_key
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.prim_name_key = 'name'
        self.prim_inputs_key = 'inputs'
        self.features_key = 'metafeatures'
        self.mask_key = 'mask'
        self.dataset_input_type = dataset_input_type

    def _get_meta_model(self, train_data):
        raise NotImplementedError()

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, validation_ratio, patience, *,
        output_dir=None, verbose=False
    ):

        # Get the mapping of primitives to their one hot encoding
        self.primitive_name_to_enc = self._get_primitive_name_to_enc(train_data=train_data)

        PyTorchModelBase.fit(
            self, train_data, n_epochs, learning_rate, batch_size, validation_ratio, patience,
            output_dir=output_dir, verbose=verbose
        )

    def _get_primitive_names(self, train_data) -> list:
        primitive_names = set()

        # Get a set of all the primitives in the train set
        for instance in train_data:
            primitives = instance[self.pipeline_key][self.steps_key]
            for primitive in primitives:
                primitive_name = primitive[self.prim_name_key]
                primitive_names.add(primitive_name)

        self.num_primitives = len(primitive_names)
        return sorted(primitive_names)

    def _get_primitive_name_to_enc(self, train_data):
        raise NotImplementedError()

    def _encode_pipeline(self, instance):
        raise NotImplementedError()

    def _yield_batch(self, group_dataloader_iters: dict, group: str):
        pipeline_batch, x_batch, y_batch = next(group_dataloader_iters[group])
        return (pipeline_batch, x_batch), y_batch

    def _pipelines_encoded(self, data) -> bool:
        pipeline_component = data[0][self.pipeline_key][self.steps_key][0]
        return type(pipeline_component) == np.ndarray

    def _get_dataset_dataloader(self, data, batch_size, drop_last, shuffle=True):
        return DAGDataLoader(
            data=data,
            group_key=self.batch_group_key,
            dataset_params={
                'features_key': self.features_key,
                'target_key': self.target_key,
                'device': self.device,
                'input_type': self.dataset_input_type,
                'mask_key': self.mask_key
            },
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=self._data_loader_seed,
            encode_pipeline=self._encode_pipeline,
            yield_batch=self._yield_batch,
            pipelines_encoded=self._pipelines_encoded
        )

    def _get_n_features(self, train_data):
        traditional_mfs_len: int = len(train_data[0][self.features_key])

        if self.metafeatures_type == 'traditional':
            return traditional_mfs_len

        deep_mfs_len: int = self.metafeature_model_config['out_h'] * self.metafeature_model_config['out_n_head']
        if self.metafeatures_type == 'deep':
            return deep_mfs_len

        if self.metafeatures_type == 'both':
            return traditional_mfs_len + deep_mfs_len
