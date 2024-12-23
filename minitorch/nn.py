from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    reshaped = reshaped.permute(0, 1, 2, 4, 3, 5)
    reshaped = reshaped.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )
    return reshaped, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute a 2D average pool to an image tensor

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    reshaped, new_height, new_width = tile(input, kernel)
    return (
        reshaped.mean(dim=4)
        .contiguous()
        .view(input.shape[0], input.shape[1], new_height, new_width)
    )


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Compute the forward pass for a max reduction."""
        ctx.save_for_backward(t1, dim)
        return t1.f.max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, float]:
        """Compute the backward pass for a max reduction."""
        t1, dim = ctx.saved_values
        return grad_output * argmax(t1, dim), 0.0


def max(a: Tensor, dim: int) -> Tensor:
    """Compute the max reduction."""
    return Max.apply(a, a._ensure_tensor(dim))


def argmax(a: Tensor, dim: int) -> Tensor:
    """Compute the argmax reduction."""
    max_val = max(a, dim)
    return a == max_val


def softmax(a: Tensor, dim: int) -> Tensor:
    """Compute the softmax reduction."""
    max_val = max(a, dim)
    exp_val = (a - max_val).exp()
    return exp_val / exp_val.sum(dim)


def logsoftmax(a: Tensor, dim: int) -> Tensor:
    """Compute the log softmax reduction."""
    max_val = max(a, dim)
    shifted = a - max_val
    exp_val = shifted.exp()
    sum_exp = exp_val.sum(dim)
    return shifted - sum_exp.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute a 2D max pool to an image tensor."""
    reshaped, new_height, new_width = tile(input, kernel)
    pool = max(reshaped, dim=4)
    return pool.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Compute a dropout reduction."""
    if ignore:
        return input
    mask = rand(input.shape) > rate
    if rate == 1.0:
        scale = 1.0
    else:
        keep_prob = 1.0 - rate
        scale = 1.0 / keep_prob
    return input * mask * scale
