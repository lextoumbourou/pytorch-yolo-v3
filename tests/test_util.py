import torch

import util


def test_predict_transform():
    """Take a detection feature map and turns it into a 2d tensor."""
    input_img = torch.rand(1, 255, 10, 10)
    cuda = False
    num_classes = 80

    anchors = [(116, 90), (156, 198), (373, 326)]

    input_dim = 320

    out = util.predict_transform(
        input_img, input_dim, anchors, num_classes, cuda)

    assert out.shape == torch.Size([1, 300, 85])
