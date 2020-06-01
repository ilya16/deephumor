"""Image captioning models."""

from torch import nn

from models import ImageEncoder, TransformerDecoder


class CaptioningTransformer(nn.Module):
    """Transformer-based image captioning model.

    - ResNet-based ImageEncoder for getting global and spacial image embeddings.
    - Vanilla Transformer Decoder.

    Global image embedding is prepended to the token embedding of decoder input sequences.
    Spacial image embeddings are used as encoder outputs in the encoder-attention block
    of the Decoder layers.
    """

    def __init__(self, num_tokens, hid_dim=512, n_layers=6, n_heads=8, pf_dim=2048,
                 enc_dropout=0.3, dec_dropout=0.1, pad_index=0, max_len=128):
        """Initializes CaptioningTransformer.

        Args:
            num_tokens (int): number of tokens in caption sequences
            hid_dim (int): hidden dimension and embedding sizes
            n_layers (int): number of Decoder layers
            n_heads (int): number of attention heads
            pf_dim (int): dimensions of the position-wise layer
            enc_dropout (float): image embeddigns dropout
            dec_dropout (float): attention and position-wise layer dropouts of the Decoder
            pad_index (int): index used for padding values in input sequences
            max_len (int): maximum lengths of input sequences.
        """

        super().__init__()

        self.image_encoder = ImageEncoder(emb_dim=hid_dim, dropout=enc_dropout)
        self.decoder = TransformerDecoder(
            num_tokens=num_tokens,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=pf_dim,
            dropout=dec_dropout,
            pad_index=pad_index,
            max_len=max_len
        )

    def forward(self, images, captions, lengths=None):
        """
        Args:
            images (torch.Tensor): input images of shape `[bs, width, height]`
            captions (torch.Tensor): text captions of shape `[bs, seq_len]`
            lengths (torch.Tensor): lengths of the input sequences of shape `[bs,]`

        Returns:
            torch.Tensor: decoded scores for caption sequence tokens of shape `[bs, seq_len, num_tokens]`
        """
        image_emb, image_spacial_emb = self.image_encoder(images)
        out = self.decoder(captions, enc_out=image_spacial_emb, image_emb=image_emb)
        out = self.classifier(out)

        return out
