from .base_models import DAGRegressionRankModelBase
from .torch_modules.lstm_mlp import LSTMMLP

import numpy as np
from torch import float32


class LSTMModel(DAGRegressionRankModelBase):

    def __init__(
        self, super_args: tuple, meta_model_activation_name: str, use_batch_norm: bool, use_skip: bool,
        hidden_state_size: int, lstm_n_layers: int, dropout: float, output_n_hidden_layers: int,
        output_hidden_layer_size: int, batch_group_key: str = 'pipeline_len'
    ):
        super().__init__(meta_model_activation_name, batch_group_key, float32, *super_args)

        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.hidden_state_size = hidden_state_size
        self.lstm_n_layers = lstm_n_layers
        self.dropout = dropout
        self.output_n_hidden_layers = output_n_hidden_layers
        self.output_hidden_layer_size = output_hidden_layer_size

    def _get_primitive_name_to_enc(self, train_data):
        primitive_names: list = self._get_primitive_names(train_data)

        # Get one hot encodings of all the primitives
        encoding = np.identity(n=self.num_primitives)

        # Create a mapping of primitive names to one hot encodings
        primitive_name_to_enc = {}
        for primitive_name, primitive_encoding in zip(primitive_names, encoding):
            primitive_name_to_enc[primitive_name] = primitive_encoding

        return primitive_name_to_enc

    def _encode_pipeline(self, instance):
        pipeline = instance[self.pipeline_key][self.steps_key]

        # Create a tensor of encoded primitives
        encoding = []
        for primitive in pipeline:
            primitive_name = primitive[self.prim_name_key]
            try:
                encoded_primitive = self.primitive_name_to_enc[primitive_name]
            except():
                raise KeyError('A primitive in this data set is not in the primitive encoding')

            encoding.append(encoded_primitive)

        instance[self.pipeline_key][self.steps_key] = encoding

    def _get_meta_model(self, train_data):
        n_features = self._get_n_features(train_data)
        return LSTMMLP(
            input_size=self.num_primitives,
            hidden_size=self.hidden_state_size,
            lstm_n_layers=self.lstm_n_layers,
            dropout=self.dropout,
            mlp_extra_input_size=n_features,
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            output_size=1,
            mlp_activation_name=self.activation_name,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            device=self.device,
            seed=self._model_seed
        )
