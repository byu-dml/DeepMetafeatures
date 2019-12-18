import typing

import torch
import torch.nn as nn

from .torch_utils import get_activation, get_reduction
from .submodule import Submodule


class DNAModule(nn.Module):

    def __init__(
        self, submodule_input_sizes: typing.Dict[str, int], n_layers: int, input_layer_size: int, hidden_layer_size: int,
        output_layer_size: int, activation_name: str, use_batch_norm: bool, use_skip: bool = False, dropout: float = 0.0, 
        reduction_name: str = 'max', *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__()
        self.submodule_input_sizes = submodule_input_sizes
        self.n_layers = n_layers
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.activation_name = activation_name
        self._activation = get_activation(activation_name, functional=True)
        self.reduction_name = reduction_name
        if self.reduction_name == 'concat':
            self.reduction = torch.cat
            self.reduction_dim = 1
        else:
            self.reduction = get_reduction(reduction_name)
            self.reduction_dim = 0
        self.reduction_name = reduction_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.device = device
        self.seed = seed
        self._input_seed = seed + 1
        self._output_seed = seed + 2
        self._dna_base_seed = seed + 3
        self._input_submodule = self._get_input_submodule()
        self._output_submodule = self._get_output_submodule()
        self._dynamic_submodules = self._get_dynamic_submodules()

    def _get_input_submodule(self):
        layer_sizes = [self.input_layer_size] + [self.hidden_layer_size] * (self.n_layers - 1)
        return Submodule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
            seed=self._input_seed
        )

    def _get_output_submodule(self):
        layer_sizes = [self.hidden_layer_size] * (self.n_layers - 1) + [self.output_layer_size]
        return Submodule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
            seed=self._output_seed
        )

    def _get_dynamic_submodules(self):
        dynamic_submodules = torch.nn.ModuleDict()
        for i, (submodule_id, submodule_input_size) in enumerate(sorted(self.submodule_input_sizes.items())):
            layer_sizes = [self.hidden_layer_size * submodule_input_size] + [self.hidden_layer_size] * (self.n_layers - 1)
            dynamic_submodules[submodule_id] = Submodule(
                layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
                seed=self._dna_base_seed + i
            )
        return dynamic_submodules

    def forward(self, args):
        pipeline_id, pipeline, x = args
        outputs = {'inputs.0': self._input_submodule(x)}
        for i, step in enumerate(pipeline['steps']):
            inputs = self._reduce([outputs[j] for j in step['inputs']])
            submodule = self._dynamic_submodules[step['name']]
            h = self._activation(submodule(inputs))
            outputs[i] = h
        return torch.squeeze(self._output_submodule(h))

    def _reduce(self, tensors: typing.Sequence[torch.Tensor]):
        if len(tensors) == 1:
            return tensors[0]
        else:
            if self.reduction_name == 'concat':
                reducible = tuple(tensors)
            else:
                reducible = torch.stack(tensors)
            reduced = self.reduction(reducible, dim=self.reduction_dim)
            return reduced
