"""Image captioning models."""

from torch import nn

from models import ImageEncoder, TransformerDecoder, LSTMDecoder, ImageLabelEncoder


class CaptioningLSTM(nn.Module):
    """LSTM-based image captioning model.

    Encodes input images into a embeddings of size `emb_dim`
    and passes them as the first token to the caption generation decoder.
    """
    def __init__(self, num_tokens, emb_dim=256, hidden_size=512, num_layers=2,
                 enc_dropout=0.3, dec_dropout=0.1):
        super(CaptioningLSTM, self).__init__()

        self.encoder = ImageEncoder(
            emb_dim=emb_dim,
            dropout=enc_dropout
        )

        self.decoder = LSTMDecoder(
            num_tokens=num_tokens,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dec_dropout,
        )

    def forward(self, images, captions, lengths=None):
        encoded = self.encoder(images)
        out = self.decoder(encoded, captions, lengths)

        return out

    def inference(self, images, temperature=2, max_seg_length=25, k=10, most_probable=True):
        encoded = self.encoder(images)

        sampled_ids = self.decoder.inference(
            encoded, temperature=temperature, max_seg_length=max_seg_length,
            k=k, most_probable=most_probable
        )

        return sampled_ids


class CaptioningLSTMWithLabels(nn.Module):
    """LSTM-based image captioning model with label inputs.

    Uses image and text label to condition the decoder.

    Encoder build combined embeddings of size `emb_dim` for input images and text labels
    and passes them as the first token to the caption generation decoder.
    """
    def __init__(self, num_tokens, emb_dim=256, hidden_size=512, num_layers=2,
                 enc_dropout=0.3, dec_dropout=0.1):
        super(CaptioningLSTMWithLabels, self).__init__()

        self.encoder = ImageLabelEncoder(
            num_tokens=num_tokens,
            emb_dim=emb_dim,
            dropout=enc_dropout
        )

        self.decoder = LSTMDecoder(
            num_tokens=num_tokens,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dec_dropout,
        )

    def forward(self, images, captions, lengths, labels):
        encoded = self.encoder(images)
        out = self.decoder(encoded, captions, lengths, labels)

        return out

    def inference(self, images, temperature=2, max_seg_length=25, k=10, most_probable=True):
        encoded = self.encoder(images)

        sampled_ids = self.decoder.inference(
            encoded, temperature=temperature, max_seg_length=max_seg_length,
            k=k, most_probable=most_probable
        )

        return sampled_ids


class CaptioningTransformer(nn.Module):
    """Transformer-based image captioning model.

    - ResNet-based [1] ImageEncoder for getting global and spacial image embeddings.
    - Vanilla Transformer Decoder [2].

    Global image embedding is prepended to the token embedding of decoder input sequences.
    Spacial image embeddings are used as encoder outputs in the encoder-attention block
    of the Decoder layers.

    References:
        [1]: "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
        [2]: "Attention Is All You Need", https://arxiv.org/abs/1706.03762
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

        self.encoder = ImageEncoder(
            emb_dim=hid_dim,
            dropout=enc_dropout,
            spacial_features=True
        )

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
        image_emb, image_spacial_emb = self.encoder(images)
        out = self.decoder(captions, enc_out=image_spacial_emb, image_emb=image_emb)
        out = self.classifier(out)

        return out
