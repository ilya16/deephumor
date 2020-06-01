"""Image and Text Encoder models."""
import torch
from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    """ResNet-based [1] image encoder.

    Encodes an image into a `emb_size` vector.

    If `spacial_features=True`, encoder also builds spacial features
    of the image based on the output of the last block of ResNet.
    The shape of spacial features is `[k x k, emb_size]`

    Note: `nn.Linear` layer is shared for global and spacial encodings.

    References:
        [1]: "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
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
        """
        Args:
            images (torch.Tensor): input images of shape `[bs, width, height]`

        Returns:
            torch.Tensor: global image embedding of shape `[bs, emb_dim]` if `self.spacial_features=False`,
                (`self.spacial_features=True`) spacial image embeddings of shape `[bs, k_w x k_h, emb_dim]`
        """
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


class LabelEncoder(nn.Module):
    """Label encoder.

    Encodes text labels into a single embedding of size `emb_dim`.

    Label Encoder 2 from [1].

    References:
        [1]: "Dank Learning: Generating Memes Using Deep Neural Networks", https://arxiv.org/abs/1806.04510
    """

    def __init__(self, num_tokens, emb_dim=256, dropout=0.2):
        """Initializes LabelEncoder.

        Args:
            num_tokens: number of tokens in the vocabulary
            emb_dim (int): dimensions of the output embedding
            dropout (float): dropout for the encoded features
        """
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, labels):
        """
        Args:
            labels (torch.Tensor): input text labels of shape `[bs, seq_len]`

        Returns:
            torch.Tensor: average label embedding of shape `[bs, emb_dim]`
        """
        emb = self.embedding(labels).mean(dim=1)
        emb = self.dropout(emb)
        return emb


class ImageLabelEncoder(nn.Module):
    """ImageLabel encoder.

    Encodes images and text labels into a single embedding of size `emb_dim`.
    """

    def __init__(self, num_tokens, emb_dim=256, dropout=0.2):
        """Initializes LabelEncoder.

        Args:
            num_tokens: number of tokens in the vocabulary
            emb_dim (int): dimensions of the output embedding
            dropout (float): dropout for the encoded features
        """
        super().__init__()
        self.image_encoder = ImageEncoder(emb_dim, dropout)
        self.label_encoder = LabelEncoder(num_tokens, emb_dim, dropout)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images, labels):
        """
        Args:
            images (torch.Tensor): input images of shape `[bs, width, height]`
            labels (torch.Tensor): input text labels of shape `[bs, seq_len]`

        Returns:
            torch.Tensor: combined image-label embedding of shape `[bs, emb_dim]`
        """
        image_emb = self.image_encoder(images)
        label_emb = self.label_encoder(labels)

        emb = torch.cat([image_emb, label_emb], dim=1)
        emb = self.dropout(self.linear(emb))

        return emb
