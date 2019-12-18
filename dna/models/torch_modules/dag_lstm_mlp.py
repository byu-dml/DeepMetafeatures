import torch
import torch.nn as nn

from .dag_lstm import DAGLSTM
from .submodule import Submodule


class DAGLSTMMLP(nn.Module):
    """
    A DAG LSTM MLP embeds a DAG, concatenates that embedding with another feature vector, and passes the concatenated
    features to an MLP.
    """

    def __init__(
            self, lstm_input_size: int, lstm_hidden_state_size: int, lstm_n_layers: int, dropout: float,
            mlp_extra_input_size: int, mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, mlp_activation_name: str,
            output_size: int, mlp_use_batch_norm: bool, mlp_use_skip: bool, reduction_name: str, *, device: str, seed: int,
    ):
        super().__init__()

        self.device = device
        self.seed = seed
        self._lstm_seed = seed + 1
        self._mlp_seed = seed + 2

        self._dag_lstm = DAGLSTM(
            lstm_input_size, lstm_hidden_state_size, lstm_n_layers, dropout, reduction_name, device=self.device,
            seed=self._lstm_seed
        )
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.lstm_n_layers = lstm_n_layers

        mlp_input_size = lstm_hidden_state_size + mlp_extra_input_size
        mlp_layer_sizes = [mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [output_size]
        self._mlp = Submodule(
            mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout, device=self.device,
            seed=self._mlp_seed,
        )

    def forward(self, args):
        (dag_structure, dags, features) = args

        batch_size = dags.shape[0]
        assert len(features) == batch_size, 'DAG batch size does not match features batch size'

        hidden_state = torch.zeros(
            self.lstm_n_layers, batch_size, self.lstm_hidden_state_size, device=self.device
        )

        dag_rnn_output = self._dag_lstm(dags, dag_structure, (hidden_state, hidden_state))
        fc_input = torch.cat((dag_rnn_output, features), dim=1)

        return self._mlp(fc_input).squeeze()
