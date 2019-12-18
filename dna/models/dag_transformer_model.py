from .transformer_model import TransformerModel
from .torch_modules.dag_transformer import DAGTransformer

from numpy import full


class DAGTransformerModel(TransformerModel):

    def _encode_pipeline(self, instance):
        pipeline = instance[self.pipeline_key][self.steps_key]
        dag_structure = [primitive[self.prim_inputs_key] for primitive in pipeline]
        pipeline_len = len(dag_structure)
        mask = full((pipeline_len, pipeline_len), -float('inf'))

        for i, inputs in enumerate(dag_structure):
            mask[i, i] = 0.0
            for j in inputs:
                if j == 'inputs.0':
                    break
                mask[i, j] = 0.0

        instance[self.pipeline_key][self.mask_key] = mask

        super()._encode_pipeline(instance)

    def _yield_batch(self, group_dataloader_iters: dict, group: str):
        mask_batch, pipeline_batch, x_batch, y_batch = next(group_dataloader_iters[group])
        return (mask_batch, pipeline_batch, x_batch), y_batch

    def _get_meta_model(self, train_data):
        n_features = self._get_n_features(train_data)
        return DAGTransformer(
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
