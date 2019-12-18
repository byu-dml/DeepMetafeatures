from .base_models import DAGRegressionRankModelBase
from .torch_modules.transformer import Transformer

from torch import int64
from numpy import array


class TransformerModel(DAGRegressionRankModelBase):
    def __init__(
        self, super_args: tuple, meta_model_activation_name: str, embedding_dim: int, n_head: int,
        n_encoder_layers: int, dim_feedforward: int, n_decoder_layers: int
    ):
        super().__init__(meta_model_activation_name, 'pipeline_len', int64, *super_args)

        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_encoder_layers = n_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.n_decoder_layers = n_decoder_layers

    def _get_primitive_name_to_enc(self, train_data):
        primitive_names = self._get_primitive_names(train_data)

        prim_to_enc: dict = {}
        for i, primitive_name in enumerate(primitive_names, start=1):
            prim_to_enc[primitive_name] = i

        return prim_to_enc

    def _encode_pipeline(self, instance):
        pipeline = instance[self.pipeline_key][self.steps_key]

        indices1 = []
        indices2 = []
        indices3 = []
        for primitive in pipeline:
            primitive_name = primitive[self.prim_name_key]
            prim_idx1: int = self.primitive_name_to_enc[primitive_name]
            indices1.append(prim_idx1)

            inputs: list = primitive[self.prim_inputs_key]
            null_prim_idx: int = 0

            prim_input1 = inputs[0]
            if prim_input1 == 'inputs.0':
                # If this primitive has no inputs, assign both its inputs the null primitive
                indices2.append(null_prim_idx)
                indices3.append(null_prim_idx)
            else:
                # Get the first input to this primitive
                prim_idx2: int = indices1[prim_input1]
                indices2.append(prim_idx2)

                if len(inputs) > 1:
                    prim_input2 = inputs[1]
                    prim_idx3: int = indices1[prim_input2]
                    indices3.append(prim_idx3)
                else:
                    # If the primitive only has one input, assign its second input the null primitive
                    indices3.append(null_prim_idx)
        instance[self.pipeline_key][self.steps_key] = array(indices1), array(indices2), array(indices3)

    def _get_meta_model(self, train_data):
        n_features = self._get_n_features(train_data)
        return Transformer(
            activation_name=self.activation_name,
            embedding_dim=self.embedding_dim,
            n_head=self.n_head,
            n_encoder_layers=self.n_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            n_decoder_layers=self.n_decoder_layers,
            features_dim=n_features,
            n_embeddings=self.num_primitives + 1,
            device=self.device,
            seed=self._model_seed
        )
