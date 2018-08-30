import os

from torch.autograd import Variable


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
