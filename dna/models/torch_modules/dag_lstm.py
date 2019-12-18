import torch
import torch.nn as nn

from .torch_utils import get_reduction
from dna.models.torch_modules.torch_utils import PyTorchRandomStateContext


class DAGLSTM(nn.Module):
    """
    A modified LSTM that handles directed acyclic graphs (DAGs) by reusing and aggregating hidden states.
    """

    def __init__(
            self, input_size: int, hidden_size: int, n_layers: int, dropout: float, reduction_name: str, *, device: str,
            seed: int
    ):
        super().__init__()

        self.reduction = get_reduction(reduction_name)
        self.reduction_dim = 0

        if dropout > 0:
            # Disable cuDNN so that the LSTM layer is deterministic, see https://github.com/pytorch/pytorch/issues/18110
            torch.backends.cudnn.enabled = False

        with PyTorchRandomStateContext(seed=seed):
            # TODO: use LSTMCell
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout
            )
        self.lstm.to(device=device)

    def forward(self, dags, dag_structure, lstm_start_state):
        assert len(dag_structure) > 0
        assert len(dag_structure) == dags.shape[1]

        prev_lstm_states = {'inputs.0': lstm_start_state}  # TODO: use something better than 'inputs.0'
        for i, node_inputs in enumerate(dag_structure):
            # TODO: use LSTMCell
            # uses lstm with sequence of len 1
            nodes_i = dags[:, i, :].unsqueeze(dim=1)  # sequence of len 1
            lstm_input_state = self._get_lstm_state(prev_lstm_states, node_inputs)
            lstm_output, lstm_output_state = self.lstm(nodes_i, lstm_input_state)
            prev_lstm_states[i] = lstm_output_state

        return lstm_output.squeeze(dim=1)

    def _get_lstm_state(self, prev_lstm_states, node_inputs):
        """
        Computes the aggregate hidden state for a node in the DAG.
        """
        mean_hidden_state = self.reduction(
            torch.stack([prev_lstm_states[i][0] for i in node_inputs]), dim=self.reduction_dim
        )
        mean_cell_state = self.reduction(
            torch.stack([prev_lstm_states[i][1] for i in node_inputs]), dim=self.reduction_dim
        )
        return (mean_hidden_state, mean_cell_state)
