import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from dna.models.torch_modules.torch_utils import PyTorchRandomStateContext, get_activation
from dna.models.model_utils import OptimizeCommand, PredictionsTargetsCommand


class MetafeatureModel(nn.Module):

    # The transformer dropout must be 0.0 so it can be deterministic
    dropout: float = 0.0

    def __init__(
            self, inp_dim: int, col_h: int, col_n_head: int, row_h: int, row_n_head: int, out_h: int,
            out_n_head: int, col_num_encoder_layers: int, col_num_decoder_layers: int, col_dim_feedforward: int,
            row_num_encoder_layers: int, row_num_decoder_layers: int, row_dim_feedforward: int,
            out_num_encoder_layers: int, out_num_decoder_layers: int, out_dim_feedforward: int,
            metafeature_model_activation_name: str, max_subset_size: int, compressor_name: str, min_n_subsets: int,
            seed: int
    ):
        """
        :param inp_dim: The size of the vectors which correspond to column features before they are projected
        :param col_h: The head size of those column feature vectors after they are projected and divided into heads
        :param col_n_head: The number of heads in the attention mechanisms of the column transformer
        :param row_h: The size of the vector representation of each row in the data set when they are divided into heads
        :param row_n_head: The number of heads in the attention mechanisms of the row transformer
        :param out_h: The size of the heads that make up the final metafeature vector
        :param out_n_head: The number of heads in the attention mechanisms of the output transformer
        :param col_num_encoder_layers: The number of encoder layers in the column transformer
        :param col_num_decoder_layers: The number of decoder layers in the column transformer
        :param col_dim_feedforward: The size of the feed forward layer in the column transformer
        :param row_num_encoder_layers: The number of encoder layers in the row transformer
        :param row_num_decoder_layers: The number of decoder layers in the row transformer
        :param row_dim_feedforward: The size of the feed forward layer in the row transformer
        :param out_num_encoder_layers: The number of encoder layers in the output transformer
        :param out_num_decoder_layers: The number of decoder layers in the output transformer
        :param out_dim_feedforward: The size of the feed forward layer in the output transformer
        :param activation_name: Name of the activation function to use
        :param max_subset_size: The sample size of each data set
        :param compressor_name: The name of the sequence compressor to use
        :param seed: The seed for initializing the metafeature model's weights
        """

        super(MetafeatureModel, self).__init__()

        self.max_subset_size = max_subset_size
        self.min_n_subsets = min_n_subsets
        dropout = MetafeatureModel.dropout

        # The size of the vectors of the sequences must be a multiple of the number of heads which they divide into
        col_dim: int = col_h * col_n_head
        row_dim: int = row_h * row_n_head
        out_dim: int = out_h * out_n_head

        with PyTorchRandomStateContext(seed=seed):
            self.activation: nn.Module = get_activation(metafeature_model_activation_name)()

            self.feat_col_projection = nn.Linear(inp_dim, col_dim)
            self.targ_col_projection = nn.Linear(inp_dim, col_dim)

            self.col_transformer = nn.Transformer(
                d_model=col_dim, nhead=col_n_head, num_encoder_layers=col_num_encoder_layers,
                dim_feedforward=col_dim_feedforward, num_decoder_layers=col_num_decoder_layers, dropout=dropout
            )

            self.feat_row_projection = nn.Linear(col_dim, row_dim)
            self.targ_row_projection = nn.Linear(col_dim, row_dim)

            self.targ_row_compressor = self.get_compressor(
                compressor_name=compressor_name, item_size=row_dim, n_heads=row_n_head, dropout=dropout,
                activation=self.activation
            )

            self.row_transformer = nn.Transformer(
                d_model=row_dim, nhead=row_n_head, num_encoder_layers=row_num_encoder_layers,
                dim_feedforward=row_dim_feedforward, num_decoder_layers=row_num_decoder_layers, dropout=dropout
            )

            self.feat_out_projection = nn.Linear(row_dim, out_dim)
            self.targ_out_projection = nn.Linear(row_dim, out_dim)
            self.out_projection = nn.Linear(2 * out_dim, out_dim)

            self.out_transformer = nn.Transformer(
                d_model=out_dim, nhead=out_n_head, num_encoder_layers=out_num_encoder_layers,
                dim_feedforward=out_dim_feedforward, num_decoder_layers=out_num_decoder_layers, dropout=dropout
            )

            self.targ_out_compressor = self.get_compressor(
                compressor_name=compressor_name, item_size=out_dim, n_heads=out_n_head, dropout=dropout,
                activation=self.activation
            )

    def single_pass(self, dataset: Tensor):
        # Separate the features from the targets
        y = dataset[-1].unsqueeze(dim=0)  # [1, rows, inp dim]
        x = dataset[:-1]  # [columns-1, rows, inp dim]

        # Project the features and the targets to the size of the column vectors
        x = self.activation(self.feat_col_projection(x))  # [columns-1, rows, col dim]
        y = self.activation(self.targ_col_projection(y))  # [1, rows, col dim]

        # Attend across the columns using the number of rows in the subset as the batch size
        x = self.col_transformer(x, y)  # [1, rows, col dim]
        x = self.activation(x)

        # Now make the batch size equal to 1 and make the number of rows the sequence length
        x = x.permute(1, 0, 2)  # [rows, 1, col dim]
        y = y.permute(1, 0, 2)  # [rows, 1, col dim]

        # Project the row representation of the targets and features to the size of the row vectors
        x = self.activation(self.feat_row_projection(x))  # [rows, 1, row dim]
        y = self.activation(self.targ_row_projection(y))  # [rows, 1, row dim]

        # Compress the targets into a single row target vector
        y = self.targ_row_compressor(y)  # [1, 1, row dim]

        # Attend across the rows and combine them into a single vector using the row target vector
        x = self.row_transformer(x, y)  # [1, 1, row dim]
        x = self.activation(x)

        # Project the row feature and row target vectors to the output size so they can be combined later
        x = self.activation(self.feat_out_projection(x))  # [1, 1, out dim]
        y = self.activation(self.targ_out_projection(y))  # [1, 1, out dim]

        # Pass the target and feature vector through a final linear layer
        # This is so the gradients can make their way to the targ_out_projection so it can actually learn
        x = torch.cat([x, y], dim=-1)  # [1, 1, 2 * out_dim]
        x = self.out_projection(x)  # [1, 1, out_dim]

        return x, y

    def forward(
        self, dataset, optimize_command: OptimizeCommand = None,
        predictions_targets_command: PredictionsTargetsCommand = None
    ):
        """
        :param dataset: [columns, rows, inp dim] The dataset to extract metafeatures from
        :param optimize_command: If training, the command to optimize the preliminary metafeatures to save memory
        :param predictions_targets_command: If training, the command to produce the y and y_hat to optimize with
        :return: [out dim] The deep metafeatures after combining the preliminary metafeatures into a final vector
        """

        # Divide the dataset into subsets of a maximum size to ensure we don't overflow our memory
        row_dim = 1
        n_rows = dataset.shape[row_dim]
        n_subsets: int = max(self.min_n_subsets, int(np.ceil(n_rows / self.max_subset_size)))
        subsets: tuple = dataset.chunk(n_subsets, dim=row_dim)

        if self.training:
            # In order to save memory that's produced from gradients, create preliminary meta features and optimize
            # The optimization will free the gradients and will teach the model to compress subsets into metafeatures
            # These subset metafeatures will then be combined by the output transformer to produce final metafeatures
            compressed_subsets = self.compress_subsets_free_gradients(
                subsets, predictions_targets_command, optimize_command
            )
        else:
            # Since we are not storing gradients, just get all the subset metafeatures without optimizing
            compressed_subsets = [self.single_pass(subset) for subset in subsets]

        # Concatenate all the preliminary metafeatures that came from the subsets
        x = torch.cat([x for x, y in compressed_subsets], dim=0)  # [n_subsets, 1, out dim]
        y = torch.cat([y for x, y in compressed_subsets], dim=0)  # [n_subsets, 1, out dim]

        # Create metafeatures for the entire dataset by using attention to combine the subset metafeatures
        y = self.targ_out_compressor(y)  # [1, 1, out dim]
        metafeatures = self.out_transformer(x, y).squeeze()  # [out dim]

        return metafeatures

    def compress_subsets_free_gradients(
        self, subsets, predictions_targets_command: PredictionsTargetsCommand, optimize_command: OptimizeCommand
    ):
        """
        The purpose of this function is to compress the subsets of a dataset into preliminary metafeatures that can be
        combined by the metafeature model's output transformer. However, this function, only useful when training, will
        also use these preliminary or subset metafeatures to optimize the model as well by computing predictions and
        using targets to compute a loss. After optimizing on the loss, the gradients are freed from memory so that they
        don't accumulate as we process more subsets, causing memory to eventually run out. This way, we can have an
        indefinite number of subsets and theoretically convert a tabular dataset of any size into metafeatures.
        """

        compressed_datasets = []
        for subset in subsets:
            # Get the compressed representation of the subset features and targets
            feat_compressed, targ_compressed = self.single_pass(subset)

            # Get the predictions from the meta model using the preliminary metafeatures of the subset
            # Get the targets as well
            y_hat, y = predictions_targets_command.run(feat_compressed.squeeze())

            # Optimize using the predictions and the targets to save space by freeing the gradients
            optimize_command.run(y_hat, y)

            compressed_dataset = feat_compressed.detach(), targ_compressed.detach()
            compressed_datasets.append(compressed_dataset)
        return compressed_datasets

    @staticmethod
    def get_compressor(
            compressor_name: str, item_size: int, n_heads: int, dropout: float, activation: nn.Module = None
    ) -> nn.Module:
        kwargs: dict = {
            'activation': activation,
            'item_size': item_size
        }

        if compressor_name == 'sumnorm':
            return SumNormCompressor(**kwargs)
        elif compressor_name == 'attention':
            kwargs['n_heads'] = n_heads
            kwargs['dropout'] = dropout
            return AttentionCompressor(**kwargs)
        elif compressor_name == 'mean':
            return MeanCompressor(**kwargs)
        else:
            raise ValueError('No valid compression technique provided. Got {}'.format(compressor_name))


class SequenceCompressor(nn.Module):
    """
    The sequence compressor is the base class to various sequence compression methods. It takes in a sequence of vectors
    and compresses them to a single vector.
    """

    def __init__(self, activation: nn.Module = None):
        """
        :param activation: (optional) The activation function to use after the compression
        """

        super().__init__()

        self.seq_len_dim = 0
        self.activation = activation

    def apply_activation(self, compressed_sequence):
        """
        :param compressed_sequence: The compressed-sequence vector to activation or let alone
        :return: The activated sequence or identity
        """

        if self.activation is not None:
            return self.activation(compressed_sequence)
        return compressed_sequence

    def forward(self, sequence):
        """
        :param sequence: The sequence to compress to a single vector [seq_len, batch_size, item_size]
        :return: The compressed sequence [1, batch_size, item_size]
        """

        raise NotImplementedError


class SumNormCompressor(SequenceCompressor):
    """
    The sum normalization compressor sums the items in a sequence, normalizes the sum, and passes the resulting vector
    through a linear layer.
    """

    def __init__(self, item_size: int, **kwargs):
        """
        :param item_size: The length of the vectors that make up the sequence
        :param kwargs: Remaining arguments for the base class
        """

        super().__init__(**kwargs)

        self.item_size_dim = -1
        self.linear = nn.Linear(item_size, item_size)

    def normalize(self, compressed_sequence):
        """
        :param compressed_sequence: The compressed-sequence vector to normalize [1, batch_size, item_size]
        :return: The compressed-sequence vector normalized after being summed
        """

        # Normalize across the item's dimension since we can't guarantee that the batch size is greater than 1
        dim = self.item_size_dim
        return (compressed_sequence - compressed_sequence.mean(dim)) / compressed_sequence.std(dim)

    def forward(self, sequence):
        compressed_sequence = sequence.sum(dim=self.seq_len_dim, keepdim=True)  # [1, batch_size, item_size]
        compressed_sequence = self.normalize(compressed_sequence)
        compressed_sequence = self.linear(compressed_sequence)

        return self.apply_activation(compressed_sequence)


class AttentionCompressor(SequenceCompressor):
    """
    The attention compressor takes in a sequence and uses attention to compress it to a single vector. It does this by
    scoring each vector in the sequence by its importance and performing a weighted sum using those scores.
    """

    def __init__(self, item_size: int, n_heads: int, dropout: float, **kwargs):
        """
        :param item_size: The length of the vectors that make up the sequence
        :param n_heads: The number of heads for the multi head attention mechanism
        :param dropout: The dropout probability for the attention layer
        :param kwargs: Remaining arguments for the base class
        """

        super().__init__(**kwargs)

        self.compression_attention = nn.MultiheadAttention(item_size, n_heads, dropout=dropout)

    def forward(self, sequence):
        target = sequence.mean(dim=self.seq_len_dim, keepdim=True)  # [1, batch_size, item_size]
        compressed_sequence: tuple = self.compression_attention(
            target, sequence, sequence, need_weights=False
        )

        # Multi head attention returns a tuple, the first element of which is the output tensor that we need
        compressed_sequence = compressed_sequence[0]  # [1, batch_size, item_size]

        return self.apply_activation(compressed_sequence)


class MeanCompressor(SequenceCompressor):
    """
    The mean compressor simply means the items in the sequence and passes the resulting vector through a linear layer.
    """

    def __init__(self, item_size: int, **kwargs):
        """
        :param item_size: The length of the vectors that make up the sequence
        :param kwargs: Remaining arguments for the base class
        """

        super().__init__(**kwargs)

        self.linear = nn.Linear(item_size, item_size)

    def forward(self, sequence):
        compressed_sequence = sequence.mean(self.seq_len_dim, keepdim=True)  # [1, batch_size, item_size]
        compressed_sequence = self.linear(compressed_sequence)

        return self.apply_activation(compressed_sequence)
