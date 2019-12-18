from .torch_utils import get_activation, PyTorchRandomStateContext

from torch import nn, cat, Tensor, ones_like
from torch.nn import Parameter


class Transformer(nn.Module):
    def __init__(
        self, activation_name: str, embedding_dim: int, n_head: int, n_encoder_layers: int, dim_feedforward: int,
        n_decoder_layers: int, features_dim: int, n_embeddings, device: str, seed: int
    ):
        super().__init__()

        embedding_dim = embedding_dim * n_head
        d_model: int = embedding_dim * 3

        with PyTorchRandomStateContext(seed=seed):
            self.activation = get_activation(activation_name)()
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=n_head, num_encoder_layers=n_encoder_layers, dim_feedforward=dim_feedforward,
                num_decoder_layers=n_decoder_layers, dropout=0.0
            )
            self.features_proj = nn.Linear(features_dim, d_model)
            self.out = nn.Linear(d_model, 1)
            self.embeddings = nn.Embedding(n_embeddings, embedding_dim)

            # Zero out the null input embedding for nodes that lack inputs
            embeddings_mask: Tensor = ones_like(self.embeddings.weight)
            embeddings_mask[0] *= 0
            new_weight: Parameter = Parameter((self.embeddings.weight * embeddings_mask).detach(), requires_grad=True)
            self.embeddings.weight = new_weight
            self.to(device)

    def _get_sequences(self, sequence_indices: Tensor):
        indices1 = sequence_indices[:, 0]
        indices2 = sequence_indices[:, 1]
        indices3 = sequence_indices[:, 2]
        sequences1 = self.embeddings(indices1)
        sequences2 = self.embeddings(indices2)
        sequences3 = self.embeddings(indices3)
        sequences = cat([sequences1, sequences2, sequences3], dim=-1)
        sequences = sequences.permute(1, 0, -1)  # [sequence length, batch size, embedding size]
        return sequences

    def _get_output(self, sequence_indices, features, masks=None):
        sequences = self._get_sequences(sequence_indices)
        features = self.activation(self.features_proj(features))
        features = features.unsqueeze(0)  # [sequence length, batch size, embedding size]
        fc_input = self.activation(self.transformer(sequences, features, src_mask=masks))
        return self.out(fc_input).squeeze()

    def forward(self, args):
        sequence_indices, features = args
        return self._get_output(sequence_indices, features)
