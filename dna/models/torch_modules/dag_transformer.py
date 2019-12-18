from .transformer import Transformer

from torch import cat


class DAGTransformer(Transformer):
    def __init__(
            self, activation_name: str, embedding_dim: int, n_head: int, n_encoder_layers: int, dim_feedforward: int,
            n_decoder_layers: int, features_dim: int, n_embeddings, device: str, seed: int
    ):
        super().__init__(
            activation_name, embedding_dim, n_head, n_encoder_layers, dim_feedforward, n_decoder_layers, features_dim,
            n_embeddings, device, seed
        )

        self.n_head = n_head

    def forward(self, args):
        masks, sequence_indices, features = args

        # Make sure each head gets the same mask and that the mask is the same size as the attention weights
        batch_size: int = masks.shape[0]
        seq_len: int = masks.shape[1]
        masks = masks.transpose(1, 0)  # [seq len, batch size, seq len]
        masks = cat([masks] * self.n_head, dim=-1)  # [seq len, batch size, seq len * n_head]
        masks = masks.contiguous().view(
            seq_len, batch_size * self.n_head, seq_len
        )  # [seq len, batch size * n_head, seq len]
        masks = masks.transpose(1, 0)  # [batch size * n_head, seq len, seq len]

        return self._get_output(sequence_indices, features, masks=masks)
