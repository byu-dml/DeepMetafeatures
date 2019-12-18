from torch.nn import Module
from torch import Tensor

from dna.models.model_utils import OptimizeCommand, PredictionsTargetsCommand
from dna.models.torch_modules.metafeature_model import MetafeatureModel


class MetaModel(Module):

    def __init__(
        self, meta_model: Module, device: str, seed, metafeatures_type: str, metafeature_model_args: dict
    ):
        super().__init__()

        self.meta_model: Module = meta_model
        self.metafeatures_type: str = metafeatures_type

        self.metafeature_model = None
        if metafeatures_type == 'deep' or metafeatures_type == 'both':
            self.metafeature_model: Module = MetafeatureModel(**metafeature_model_args, seed=seed)
            self.metafeature_model.to(device)

    def forward(self, dataset_data_loader, dataset: Tensor, optimize_command: OptimizeCommand = None):
        predictions_targets_command = PredictionsTargetsCommand(
            dataset_data_loader, self.meta_model, self.metafeatures_type
        )
        dataset_vector = None

        # If using deep metafeatures, convert the tabular dataset into a vector
        if self.metafeatures_type == 'both' or self.metafeatures_type == 'deep':
            assert self.metafeature_model is not None

            if self.training:
                # If we are training, convert the data set to a vector while optimizing on the subsets to save memory
                dataset_vector = self.metafeature_model(
                    dataset, optimize_command=optimize_command, predictions_targets_command=predictions_targets_command
                )
            else:
                dataset_vector = self.metafeature_model(dataset)

        y_hat, y = predictions_targets_command.run(dataset_vector)
        return y_hat, y
