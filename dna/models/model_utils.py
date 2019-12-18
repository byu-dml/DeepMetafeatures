from torch import Tensor, log, cat, argsort, exp, softmax, sigmoid


def likelihood_loss(y_hat: Tensor, y: Tensor):
    true_ranking: Tensor = argsort(y, descending=True)
    y_hat: Tensor = y_hat[true_ranking]

    y_hat: Tensor = softmax(y_hat, dim=0)
    y_hat: Tensor = exp(y_hat)

    losses: list = []
    n: int = len(y_hat)
    for i in range(n):
        loss = y_hat[i] / y_hat[i:n].mean()
        losses.append(loss.unsqueeze(dim=0))

    loss = cat(losses, dim=0).mean()
    loss = -log(loss)
    loss = sigmoid(loss)
    return loss


def pearson_loss(y_hat: Tensor, y: Tensor):
    covariance = ((y_hat - y_hat.mean()) * (y - y.mean())).mean()
    y_hat_std = y_hat.std()
    y_std = y.std()

    # Handle NANs produced by division by zero
    if y_hat_std.item() == 0.0:
        print('WARNING: PREDICTIONS HAVE NO VARIANCE \n{}'.format(y_hat))
        y_hat_std = y_hat_std + 1e-6
    if y_std.item() == 0.0:
        print('WARNING: TARGETS HAVE NO VARIANCE \n{}'.format(y))
        y_std = y_std + 1e-6

    coefficient = covariance / (y_hat_std * y_std)
    return ((1 - coefficient) ** 2) ** 0.5


class OptimizeCommand:

    def __init__(self, loss_function: callable, optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer

    def run(self, y_hat, y):
        # Compute loss for this dataset batch and optimize
        loss = self.loss_function(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class PredictionsTargetsCommand:

    def __init__(self, dataset_data_loader, meta_model, metafeatures_type):
        self.dataset_data_loader = dataset_data_loader
        self.meta_model = meta_model
        self.metafeatures_type = metafeatures_type

    def run(self, dataset_vector: Tensor) -> tuple:
        dataset_y_hat: list = []
        dataset_y: list = []
        for x, y in self.dataset_data_loader:
            # If using deep metafeatures, stack the dataset vector for each pipeline in the batch
            if dataset_vector is not None:
                pipeline_structure = None
                if len(x) > 2:
                    pipeline_structure, pipelines, metafeatures = x
                else:
                    pipelines, metafeatures = x

                batch_size = len(pipelines)
                stacked_dataset_vector = dataset_vector.unsqueeze(dim=0).expand(batch_size, -1)

                if self.metafeatures_type == 'both':
                    # If using both deep and traditional metafeatures, concatenate both together
                    metafeatures = cat([metafeatures, stacked_dataset_vector], dim=1)
                elif self.metafeatures_type == 'deep':
                    metafeatures = stacked_dataset_vector

                if len(x) > 2:
                    assert pipeline_structure is not None
                    x = pipeline_structure, pipelines, metafeatures
                else:
                    x = pipelines, metafeatures
            y_hat = self.meta_model(x)

            if len(y_hat.shape) < 1:
                y_hat = y_hat.unsqueeze(dim=-1)

            dataset_y_hat.append(y_hat)
            dataset_y.append(y)
        y_hat: Tensor = cat(dataset_y_hat, dim=0)
        y: Tensor = cat(dataset_y, dim=0)

        return y_hat, y
