import torch

from .base_models import PyTorchRegressionRankModelBase
from .torch_modules.dna_module import DNAModule


class DNARegressionModel(PyTorchRegressionRankModelBase):

    def __init__(
        self, n_hidden_layers: int, hidden_layer_size: int, activation_name: str, use_batch_norm: bool,
        reduction_name: str = 'max', loss_function_name: str = 'rmse', use_skip: bool = False, dropout = 0.0,  *,
        device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(y_dtype=torch.float32, device=device, seed=seed, loss_function_name=loss_function_name)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation_name = activation_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.reduction_name = reduction_name
        self.output_layer_size = 1
        self._model_seed = self.seed + 1

    def _get_model(self, train_data):
        submodule_input_sizes = {}
        for instance in train_data:
            for step in instance['pipeline']['steps']:
                submodule_input_sizes[step['name']] = len(step['inputs']) if self.reduction_name == 'concat' else 1
        self.input_layer_size = len(train_data[0]['metafeatures'])

        return DNAModule(
            submodule_input_sizes, self.n_hidden_layers + 1, self.input_layer_size, self.hidden_layer_size,
            self.output_layer_size, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout,
            self.reduction_name, device=self.device, seed=self._model_seed
        )
