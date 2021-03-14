"""Image captioning models."""
from abc import abstractmethod
from inspect import getfullargspec

import torch
from torch import nn

from . import ImageEncoder, TransformerDecoder, LSTMDecoder, ImageLabelEncoder
from .utils import get_mask_from_lengths


class _CaptioningModel(nn.Module):
    """Base class for Captioning models."""
    def __init__(self):
        super().__init__()

        self.encoder = None
        self.decode = None
        self._hp = {}

    def save(self, ckpt_path):
        """Saves the model's state and hyperparameters."""
        torch.save(
            {'model': self.state_dict(), 'hp': self._hp},
            ckpt_path
        )

    @staticmethod
    def _check_parameters(params, model_cls):
        """Checks the hyperparameters of model"""
        args = set(getfullargspec(model_cls.__init__).args)
        params = {key: value for key, value in params.items() if key in args}
        return params

    @staticmethod
    def _from_hparam_dict(params, model_cls):
        """Builds the model from the parameters dict."""
        params = _CaptioningModel._check_parameters(params, model_cls)
        return model_cls(**params)

    @staticmethod
    @abstractmethod
    def from_hparam_dict(params):
        pass

    @staticmethod
    def _from_pretrained(ckpt_path, model_cls):
        """Loads and builds the model from the checkpoint file."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = model_cls.from_hparam_dict(ckpt['hp'])
        model.load_state_dict(ckpt['model'])
        return model

    @staticmethod
    @abstractmethod
    def from_pretrained(ckpt_path):
        pass


class CaptioningLSTM(_CaptioningModel):
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

    @staticmethod
    def from_hparam_dict(params):
        """Builds the model from the parameters dict."""
        return _CaptioningModel._from_hparam_dict(params, CaptioningLSTM)

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        return _CaptioningModel._from_pretrained(ckpt_path, CaptioningLSTM)


class CaptioningLSTMWithLabels(_CaptioningModel):
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

    @staticmethod
    def from_hparam_dict(params):
        """Builds the model from the parameters dict."""
        return _CaptioningModel._from_hparam_dict(params, CaptioningLSTMWithLabels)

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        return _CaptioningModel._from_pretrained(ckpt_path, CaptioningLSTMWithLabels)


class CaptioningTransformerBase(_CaptioningModel):
    """Simple Transformer-based image captioning model without Encoder-Attention Decoder blocks.

    - ResNet-based [1] ImageEncoder for getting global and spatial image embeddings.
    - Vanilla Transformer Decoder without Encoder-Attention [2].

    Global image embedding is prepended to the token embedding of decoder input sequences.

    References:
        [1]: "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
        [2]: "Attention Is All You Need", https://arxiv.org/abs/1706.03762
    """

    def __init__(self, num_tokens, hid_dim=512, n_layers=6, n_heads=8, pf_dim=2048,
                 enc_dropout=0.3, dec_dropout=0.1, max_len=128):
        """Initializes CaptioningTransformer.

        Args:
            num_tokens (int): number of tokens in caption sequences
            hid_dim (int): hidden dimension and embedding sizes
            n_layers (int): number of Decoder layers
            n_heads (int): number of attention heads
            pf_dim (int): dimensions of the position-wise layer
            enc_dropout (float): image embeddings dropout
            dec_dropout (float): attention and position-wise layer dropouts of the Decoder
            max_len (int): maximum lengths of input sequences.
        """

        super().__init__()

        self.encoder = ImageEncoder(
            emb_dim=hid_dim,
            dropout=enc_dropout,
            spatial_features=False
        )

        self.decoder = TransformerDecoder(
            num_tokens=num_tokens,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=pf_dim,
            dropout=dec_dropout,
            max_len=max_len,
            receives_context=False
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
        image_emb = self.encoder(images)
        input_mask = get_mask_from_lengths(lengths) if lengths is not None else None
        out = self.decoder(captions, input_mask=input_mask, start_emb=image_emb)

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
        image_emb = self.encoder(image)

        sampled_ids = self.decoder.generate(
            image_emb, caption=caption,
            max_len=max_len, temperature=temperature,
            beam_size=beam_size, top_k=top_k, eos_index=eos_index
        )

        return sampled_ids

    @staticmethod
    def from_hparam_dict(params):
        """Builds the model from the parameters dict."""
        return _CaptioningModel._from_hparam_dict(params, CaptioningTransformerBase)

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        return _CaptioningModel._from_pretrained(ckpt_path, CaptioningTransformerBase)


class CaptioningTransformer(_CaptioningModel):
    """Transformer-based image captioning model.

    - ResNet-based [1] ImageEncoder for getting global and spatial image embeddings.
    - Vanilla Transformer Decoder [2].

    Global image embedding is prepended to the token embedding of decoder input sequences.
    Spatial image embeddings are used as encoder outputs in the encoder-attention block
    of the Decoder layers.

    References:
        [1]: "Deep Residual Learning for Image Recognition", https://arxiv.org/abs/1512.03385
        [2]: "Attention Is All You Need", https://arxiv.org/abs/1706.03762
    """

    def __init__(self, num_tokens, hid_dim=512, n_layers=6, n_heads=8, pf_dim=2048,
                 enc_dropout=0.3, dec_dropout=0.1, max_len=128):
        """Initializes CaptioningTransformer.

        Args:
            num_tokens (int): number of tokens in caption sequences
            hid_dim (int): hidden dimension and embedding sizes
            n_layers (int): number of Decoder layers
            n_heads (int): number of attention heads
            pf_dim (int): dimensions of the position-wise layer
            enc_dropout (float): image embeddings dropout
            dec_dropout (float): attention and position-wise layer dropouts of the Decoder
            max_len (int): maximum lengths of input sequences.
        """

        super().__init__()

        self.encoder = ImageEncoder(
            emb_dim=hid_dim,
            dropout=enc_dropout,
            spatial_features=True
        )

        self.decoder = TransformerDecoder(
            num_tokens=num_tokens,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=pf_dim,
            dropout=dec_dropout,
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
        image_emb, image_spatial_emb = self.encoder(images)
        input_mask = get_mask_from_lengths(lengths) if lengths is not None else None
        out = self.decoder(captions, input_mask=input_mask, context=image_spatial_emb, start_emb=image_emb)

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
        image_emb, image_spatial_emb = self.encoder(image)

        sampled_ids = self.decoder.generate(
            image_emb, caption=caption, context=image_spatial_emb,
            max_len=max_len, temperature=temperature,
            beam_size=beam_size, top_k=top_k, eos_index=eos_index
        )

        return sampled_ids

    @staticmethod
    def from_hparam_dict(params):
        """Builds the model from the parameters dict."""
        return _CaptioningModel._from_hparam_dict(params, CaptioningTransformer)

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        return _CaptioningModel._from_pretrained(ckpt_path, CaptioningTransformer)
