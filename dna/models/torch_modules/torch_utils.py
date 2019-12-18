import torch


def get_activation(activation_name: str, functional: bool = False):
    if activation_name == 'relu':
        if functional:
            return torch.nn.functional.relu
        else:
            return torch.nn.ReLU

    if activation_name == 'leaky_relu':
        if functional:
            return torch.nn.functional.leaky_relu
        else:
            return torch.nn.LeakyReLU

    if activation_name == 'sigmoid':
        if functional:
            return torch.nn.functional.sigmoid
        else:
            return torch.nn.Sigmoid

    if activation_name == 'tanh':
        if functional:
            return torch.nn.functional.tanh
        else:
            return torch.nn.Tanh

    raise ValueError('unknown activation name: {}'.format(activation_name))


def get_reduction(reduction_name: str):
    if reduction_name == 'mean':
        return torch.mean

    if reduction_name == 'sum':
        return torch.sum

    if reduction_name == 'prod':
        return torch.prod

    if reduction_name == 'max':
        def torch_max(input, dim, keepdim=False, out=None):
            return torch.max(input=input, dim=dim, keepdim=keepdim, out=out).values
        return torch_max

    if reduction_name == 'median':
        def torch_median(input, dim, keepdim=False, out=None):
            return torch.median(input=input, dim=dim, keepdim=keepdim, out=out).values
        return torch_median

    raise ValueError('unknown reduction name: {}'.format(reduction_name))


class PyTorchRandomStateContext:

    def __init__(self, seed):
        self.seed = seed
        self._state = None

    def __enter__(self):
        self._state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, *args):
        torch.random.set_rng_state(self._state)


class BatchNorm1d(torch.nn.Module):

    def __init__(self, size):
        super().__init__()

        self.batch_norm = torch.nn.BatchNorm1d(size)

    def forward(self, x):
        if len(x) > 1:
            return self.batch_norm(x)
        else:
            return x
