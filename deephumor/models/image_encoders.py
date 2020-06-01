"""Image and Text Encoder models."""

from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    """ResNet-based image encoder.

    Encodes an image into a `emb_size` vector.

    If `spacial_features=True`, encoder also builds spacial features
    of the image based on the output of the last block of ResNet.
    The shape of spacial features is `[k x k, emb_size]`

    Note: `nn.Linear` layer is shared for global and spacial encodings.

    References:
        "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
    """

    def __init__(self, emb_dim=256, dropout=0.2, spacial_features=False):
        """Initializes ImageEncoder.

        Args:
            emb_dim (int): dimensions of the output embedding
            dropout (float): dropout for the encoded features
            spacial_features (bool): whether compute spacial features or not
        """
        super().__init__()

        self.spacial_features = spacial_features

        resnet = models.resnet50(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad = False
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.avgpool = resnet.avgpool

        # embedding layer
        self.linear = nn.Linear(resnet.fc.in_features, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        # ResNet features
        features = self.resnet(images)
        bs, dim = features.shape[:2]

        # global image embedding
        x = self.avgpool(features).reshape(bs, -1)
        emb = self.dropout(self.bn(self.linear(x)))

        # spacial features
        if self.spacial_features:
            x = features.reshape(bs, dim, -1)
            x = x.transpose(2, 1)  # (B, D, N) -> (B, N, D)
            spacial_emb = self.dropout(self.linear(x))
            return emb, spacial_emb

        return emb
