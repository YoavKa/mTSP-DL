import os

import torch
from torch.autograd import Variable

from .common import assert_result


USE_GPU = 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] != '-1'


def to_variable(*tensors, async=False, pin_memory=False, volatile=False):
    result = []
    for tensor in tensors:
        if pin_memory:
            tensor = tensor.pin_memory()
        if not isinstance(tensor, Variable):
            tensor = Variable(tensor, volatile=volatile)
        if USE_GPU:
            tensor = tensor.cuda(async=async)
        result.append(tensor)

    if len(result) == 1:
        return result[0]
    else:
        return result


# noinspection PyUnresolvedReferences
@assert_result(torch.Tensor([[[1, 2], [3, 5], [6, 7]],
                             [[10, 11], [12, 14], [15, 16]]]),
               torch.Tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                             [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]))
def strip_main_diagonal(x):
    # x:    Float(dim_0 x dim_1 x ... dim_(k-2) x n x n)
    # res:  Float(dim_0 x dim_1 x ... dim_(k-2) x n x (n-1))

    # removes the main diagonal from x, e.g:
    # From:  0 1 2
    #        3 4 5
    #        6 7 8
    # To:    1 2
    #        3 5
    #        6 7
    # by first changing x such that the main diagonal is on the first column (while discarding the last item):
    #        0 1 2 3
    #        4 5 6 7
    # and then stripping it and reshaping the tensor

    assert x.dim() >= 2, 'can only strip main diagonal of tensor of two or more dimensions!'
    assert x.size(-1) == x.size(-2), 'strip_main_diagonal currently supports striping diagonals of square dimensions' \
                                     ' only!'
    assert x.size(-1) > 1, 'cannot strip main diagonal if the dimension size is only 1!'

    base_size = x.size()[:-2]
    n = x.size(-1)

    # merge the last two dimensions
    new_x = x.contiguous().view(*base_size, n * n)

    # remove the last item on the diagonal
    new_x = new_x.transpose(0, x.dim() - 2)[:-1].transpose(0, x.dim() - 2)

    # view x such that the first column is the diagonal
    new_x = new_x.contiguous().view(*base_size, n - 1, n + 1)

    # remove the first column
    new_x = new_x.transpose(0, x.dim() - 1)[1:].transpose(0, x.dim() - 1)

    # reshape to result
    result = new_x.contiguous().view(*base_size, n, n - 1)

    return result
