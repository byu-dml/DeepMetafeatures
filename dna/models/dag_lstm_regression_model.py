from .lstm_model import LSTMModel
from .torch_modules.dag_lstm_mlp import DAGLSTMMLP

from dna.data import group_json_objects


class DAGLSTMRegressionModel(LSTMModel):

    def __init__(
        self, super_args: tuple, meta_model_activation_name: str, use_batch_norm: bool, use_skip: bool,
        hidden_state_size: int, lstm_n_layers: int, dropout: float, reduction_name: str, output_n_hidden_layers: int,
        output_hidden_layer_size: int
    ):
        super().__init__(
            super_args, meta_model_activation_name, use_batch_norm, use_skip, hidden_state_size, lstm_n_layers, dropout,
            output_n_hidden_layers, output_hidden_layer_size, 'pipeline_structure'
        )

        self.reduction_name = reduction_name
        self.pipeline_structures = None

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, validation_ratio, patience, *,
        output_dir=None, verbose=False
    ):
        self.pipeline_structures = self._get_pipeline_structures(train_data)

        super().fit(
            train_data, n_epochs, learning_rate, batch_size, validation_ratio, patience,
            output_dir=output_dir, verbose=verbose
        )

    def _get_pipeline_structures(self, train_data):
        pipeline_structures = {}
        grouped_by_structure = group_json_objects(train_data, 'pipeline_structure')
        for group, group_indices in grouped_by_structure.items():
            index = group_indices[0]
            item = train_data[index]
            pipeline = item[self.pipeline_key][self.steps_key]
            group_structure = [primitive[self.prim_inputs_key] for primitive in pipeline]
            pipeline_structures[group] = group_structure
        return pipeline_structures

    def _yield_batch(self, group_dataloader_iters: dict, group: str):
        pipeline_batch, x_batch, y_batch = next(group_dataloader_iters[group])

        # Get the structure of the pipelines in this group so the meta model can parse the pipeline
        group_structure = self.pipeline_structures[group]

        return (group_structure, pipeline_batch, x_batch), y_batch

    def _get_meta_model(self, train_data):
        n_features = self._get_n_features(train_data)
        return DAGLSTMMLP(
            lstm_input_size=self.num_primitives,
            lstm_hidden_state_size=self.hidden_state_size,
            lstm_n_layers=self.lstm_n_layers,
            dropout=self.dropout,
            mlp_extra_input_size=n_features,
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            output_size=1,
            mlp_activation_name=self.activation_name,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            reduction_name=self.reduction_name,
            device=self.device,
            seed=self._model_seed,
        )
