import typing
from torch.cuda import is_available

from .dna_regression_model import DNARegressionModel
from .lstm_model import LSTMModel
from .dag_lstm_regression_model import DAGLSTMRegressionModel
from .transformer_model import TransformerModel
from .dag_transformer_model import DAGTransformerModel


def get_model(model_name: str, model_config: typing.Dict, seed: int, metafeatures_type: str):
    model_class = {
        'dna_regression': DNARegressionModel,
        'lstm': LSTMModel,
        'daglstm_regression': DAGLSTMRegressionModel,
        'transformer': TransformerModel,
        'dag_transformer': DAGTransformerModel
    }[model_name.lower()]

    device = 'cuda:0' if is_available() else 'cpu'
    meta_model_config = model_config['meta_model']
    metafeature_model_config = model_config['metafeature_model']
    loss_function_name = meta_model_config['loss_function_name']
    del meta_model_config['loss_function_name']
    super_args: tuple = (seed, device, metafeatures_type, metafeature_model_config, loss_function_name)

    return model_class(super_args, **meta_model_config)
