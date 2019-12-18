import torch
import torch.nn as nn

from dna.models.torch_modules.torch_utils import PyTorchRandomStateContext
from .submodule import Submodule


class LSTMMLP(nn.Module):
    """
    A standard LSTM for processing straight sequences
    """

    def __init__(
        self, input_size: int, hidden_size: int, lstm_n_layers: int, dropout: float, mlp_extra_input_size: int,
        mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, output_size: int, mlp_activation_name: str,
        mlp_use_batch_norm: bool, mlp_use_skip: bool, *, device: str, seed: int
    ):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.lstm_n_layers = lstm_n_layers

        if dropout > 0:
            # Disable cuDNN so that the LSTM layer is deterministic, see https://github.com/pytorch/pytorch/issues/18110
            torch.backends.cudnn.enabled = False

        with PyTorchRandomStateContext(seed=seed+1):
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=lstm_n_layers, batch_first=True, dropout=dropout
            )
        self.lstm.to(device=device)

        mlp_input_size = hidden_size + mlp_extra_input_size
        mlp_layer_sizes = [mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [output_size]
        self._mlp = Submodule(
            mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout, device=device,
            seed=seed+2
        )

    def forward(self, args):
        sequences, features = args
        batch_size = len(sequences)
        assert batch_size == len(features)

        hidden_state = torch.zeros(
            self.lstm_n_layers, batch_size, self.hidden_size, device=self.device
        )

        lstm_output, _ = self.lstm(sequences, (hidden_state, hidden_state))

        # Get the last output of the LSTM
        lstm_output = lstm_output[:, -1, :]

        fc_input = torch.cat((lstm_output, features), dim=1)

        return self._mlp(fc_input).squeeze()
