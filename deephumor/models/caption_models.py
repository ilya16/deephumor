"""Image captioning models."""
import torch
from torch import nn

from deephumor.models import ImageEncoder, TransformerDecoder, LSTMDecoder, ImageLabelEncoder


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

        # hyperparameters dictionary
        self._hp = {
            'num_tokens': num_tokens,
            'emb_dim': emb_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'enc_dropout': enc_dropout,
            'dec_dropout': dec_dropout,
        }

    def forward(self, images, captions, lengths=None):
        emb = self.encoder(images)
        out = self.decoder(emb, captions, lengths)

        return out

    def generate(self, image, caption=None, max_len=25,
                 temperature=1.0, beam_size=10, top_k=50, eos_index=3):
        """Generates caption for an image.

        Args:
            image (torch.Tensor): input image of shape `[1, width, height]`
            caption (torch.Tensor, optional): beginning tokens of the caption of shape `[1, seq_len]`
            max_len (int): maximum length of the caption
            temperature (float): temperature for softmax over logits
            beam_size (int): number of maintained branches at each step
            top_k (int): number of the most probable tokens to consider during sampling
            eos_index (int): index of the EOS (end-of-sequence) token

        Returns:
            torch.Tensor: generated caption tokens of shape `[1, min(output_len, max_len)]`
        """

        # get image embedding
        image_emb = self.encoder(image).unsqueeze(1)

        sampled_ids = self.decoder.generate(
            image_emb, caption=caption,
            max_len=max_len, temperature=temperature,
            beam_size=beam_size, top_k=top_k, eos_index=eos_index
        )

        return sampled_ids

    def save(self, ckpt_path):
        """Saves the model's state and hyperparameters."""
        torch.save(
            {'model': self.state_dict(), 'hp': self._hp},
            ckpt_path
        )

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        hp = ckpt['hp']

        model = CaptioningLSTM(
            num_tokens=hp['num_tokens'],
            emb_dim=hp['emb_dim'],
            hidden_size=hp['hidden_size'],
            num_layers=hp['num_layers'],
            enc_dropout=hp['enc_dropout'],
            dec_dropout=hp['dec_dropout'],
        )
        model.load_state_dict(ckpt['model'])
        return model


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
            embedding=self.encoder.label_encoder.embedding
        )

        # hyperparameters dictionary
        self._hp = {
            'num_tokens': num_tokens,
            'emb_dim': emb_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'enc_dropout': enc_dropout,
            'dec_dropout': dec_dropout,
        }

    def forward(self, images, captions, lengths, labels):
        emb = self.encoder(images=images, labels=labels)
        out = self.decoder(emb, captions, lengths)

        return out

    def generate(self, image, label, caption=None, max_len=25,
                 temperature=1.0, beam_size=10, top_k=50, eos_index=3):
        """Generates caption for an image based on the text label.

        Args:
            image (torch.Tensor): input image of shape `[1, width, height]`
            label: (torch.Tensor): text label for the image `[1, label_len]`
            caption (torch.Tensor, optional): beginning tokens of the caption of shape `[1, seq_len]`
            max_len (int): maximum length of the caption
            temperature (float): temperature for softmax over logits
            beam_size (int): number of maintained branches at each step
            top_k (int): number of the most probable tokens to consider during sampling
            eos_index (int): index of the EOS (end-of-sequence) token

        Returns:
            torch.Tensor: generated caption tokens of shape `[1, min(output_len, max_len)]`
        """

        # get image embedding
        image_emb = self.encoder(image, label).unsqueeze(1)

        sampled_ids = self.decoder.generate(
            image_emb, caption=caption,
            max_len=max_len, temperature=temperature,
            beam_size=beam_size, top_k=top_k, eos_index=eos_index
        )

        return sampled_ids

    def save(self, ckpt_path):
        """Saves the model's state and hyperparameters."""
        torch.save(
            {'model': self.state_dict(), 'hp': self._hp},
            ckpt_path
        )

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        hp = ckpt['hp']

        model = CaptioningLSTMWithLabels(
            num_tokens=hp['num_tokens'],
            emb_dim=hp['emb_dim'],
            hidden_size=hp['hidden_size'],
            num_layers=hp['num_layers'],
            enc_dropout=hp['enc_dropout'],
            dec_dropout=hp['dec_dropout'],
        )
        model.load_state_dict(ckpt['model'])
        return model


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
            enc_dropout (float): image embeddings dropout
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

        # hyperparameters dictionary
        self._hp = {
            'num_tokens': num_tokens,
            'hid_dim': hid_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'pf_dim': pf_dim,
            'enc_dropout': enc_dropout,
            'dec_dropout': dec_dropout,
            'pad_index': pad_index,
            'max_len': max_len
        }

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

        return out

    def generate(self, image, caption=None, max_len=25,
                 temperature=1.0, beam_size=10, top_k=50, eos_index=3):
        """Generates caption for an image.

        Args:
            image (torch.Tensor): input image of shape `[1, width, height]`
            caption (torch.Tensor, optional): beginning tokens of the caption of shape `[1, seq_len]`
            max_len (int): maximum length of the caption
            temperature (float): temperature for softmax over logits
            beam_size (int): number of maintained branches at each step
            top_k (int): number of the most probable tokens to consider during sampling
            eos_index (int): index of the EOS (end-of-sequence) token

        Returns:
            torch.Tensor: generated caption tokens of shape `[1, min(output_len, max_len)]`
        """

        # get image embeddings
        image_emb, image_spacial_emb = self.encoder(image)

        sampled_ids = self.decoder.generate(
            image_emb, image_spacial_emb, caption=caption,
            max_len=max_len, temperature=temperature,
            beam_size=beam_size, top_k=top_k, eos_index=eos_index
        )

        return sampled_ids

    def save(self, ckpt_path):
        """Saves the model's state and hyperparameters."""
        torch.save(
            {'model': self.state_dict(), 'hp': self._hp},
            ckpt_path
        )

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        hp = ckpt['hp']

        model =  CaptioningTransformer(
            num_tokens=hp['num_tokens'],
            hid_dim=hp['hid_dim'],
            n_layers=hp['n_layers'],
            n_heads=hp['n_heads'],
            pf_dim=hp['pf_dim'],
            enc_dropout=hp['enc_dropout'],
            dec_dropout=hp['dec_dropout'],
            pad_index=hp['pad_index'],
            max_len=hp['max_len']
        )
        model.load_state_dict(ckpt['model'])
        return model
