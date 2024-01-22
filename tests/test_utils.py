import tofina.utils as utils
import torch


def test_reverseSoftmax():
    softmax = torch.nn.Softmax(dim=0)
    X = torch.tensor([0.3, 0.5, 0.2])
    assert utils.check_equality(X, softmax(utils.softmaxInverse(X)))
