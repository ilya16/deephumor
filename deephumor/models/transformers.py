"""Transformer modules.

References:
    [1]: "Attention Is All You Need", https://arxiv.org/abs/1706.03762
"""
from inspect import isfunction

import torch
from torch import nn

from deephumor.models.beam import BeamSearchHelper


def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def to(t):
    return {'device': t.device, 'dtype': t.dtype}


def get_mask_from_lengths(lengths):
    """Computes padding mask from a tensor of input lengths.

    Args:
        lengths (torch.Tensor): lengths of inputs in a batch of shape `[bs]`

    Returns:
        torch.Tensor: boolean padding mask of shape `[bs, max_seq_len]`
    """

    ids = torch.arange(lengths.max(), device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask


def get_causal_mask(seq):
    """Returns autoregressive mask for the decoder inputs.

    Args:
        seq (torch.Tensor): input sequences of shape `[bs, seq_len]`

    Returns:
        torch.bool: boolean mask of shape `[bs, seq_len, seq_len]`
    """
    ticker = torch.arange(seq.size(1), device=seq.device)
    return (ticker[None, :] > ticker[:, None]).unsqueeze(0)


class MultiHeadAttentionLayer(nn.Module):
    """MultiHeadAttentionLayer from "Attention Is All You Need"."""

    def __init__(self, hid_dim=512, n_heads=8, dropout=0., causal=False):
        """Initializes MultiHeadAttentionLayer.

        Dimension of one head is `hid_dim` // `n_heads`

        Args:
            hid_dim (int): hidden dimension size
            n_heads (int): number of attention heads
            dropout (float): attention dropout
            causal (bool): whether to apply causal mask or not
        """

        super().__init__()

        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.causal = causal

        # query, key and value linear networks
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        # output linear networks
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        # attention dropout
        self.dropout = nn.Dropout(dropout)

        # scale parameter
        self.scale = torch.nn.Parameter(
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)),
            requires_grad=False
        )

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        """
        Args:
            query (torch.Tensor): queries of shape `[bs, query_len, hid_dim]`
            key (torch.Tensor): keys of shape `[bs, key_len, hid_dim]`
            value (torch.Tensor): values of shape `[bs, key_len, hid_dim]`
            query_mask (torch.Tensor): boolean query mask of shape `[bs, query_len]`
            key_mask (torch.Tensor): boolean key mask of shape `[bs, key_len]`

        Returns:
            torch.Tensor: multi-head attention tensor of shape `[bs, seq_len, hid_dim]`
        """
        bs, query_len, key_len = *query.shape[:2], key.shape[1]
        device = query.device

        # calculate Q, K, V using corresponding linear networks
        q, k, v = self.fc_q(query), self.fc_k(key), self.fc_v(value)  # shape is [bs, seq_len, hid_dim]

        # prepare Q, K, V for .matmul() or `@` operator
        # shape is [bs, n_heads, seq_len, head_dim]
        split_heads = lambda x_: x_.view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split_heads, (q, k, v))

        # compute energy
        energy = (q @ k.transpose(2, 3)) / self.scale  # shape is [bs, n_heads, seq_q_len, seq_k_len]

        # apply masks
        mask_value = max_neg_value(energy)

        if query_mask is not None or key_mask is not None:
            q_mask = default(query_mask, lambda: torch.ones((bs, query_len), dtype=torch.bool, device=device))
            kv_mask = default(key_mask, lambda: torch.ones((bs, key_len), dtype=torch.bool, device=device))

            mask = q_mask[:, None, :, None] * kv_mask[:, None, None, :]

            energy.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            ts, kv_ts = torch.arange(query_len, **to(q)), torch.arange(key_len, **to(q))
            ts, kv_ts = ts[None, None, :], kv_ts[None, None, :]

            mask = ts[:, :, :, None] < kv_ts[:, :, None, :]
            energy.masked_fill_(mask, mask_value)
            del mask

        # apply softmax along the last dim of energy and get the attention weights
        # shape is [bs, n_heads, seq_len, seq_len]
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # weight values with calculated attention
        # shape is [bs, n_heads, seq_len, head_dim]
        x = attention @ v

        # squash 1 and 4 dims back
        x = x.transpose(1, 2).contiguous()
        x = x.view(bs, -1, self.hid_dim)  # shape is [bs, seq_len, hid_dim]

        # apply output linear layer
        x = self.fc_o(x)

        return x


class PositionwiseFeedforwardLayer(nn.Module):
    """Position-wise Feedforward Layer from "Attention Is All You Need"."""

    def __init__(self, hid_dim=512, pf_dim=2048, dropout=0.):
        """Initializes PositionwiseFeedforwardLayer.

        Args:
            hid_dim (int): hidden dimension size
            pf_dim (int): dimensions of the position-wise layer
            dropout (float): position-wise layer dropout
        """

        super().__init__()

        # linear layers
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        # dropout is applied after the first layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): sequences of shape `[bs, seq_len, hid_dim]`

        Returns:
            torch.Tensor: processed sequences of shape `[bs, seq_len, hid_dim]`
        """
        # apply linear layers + dropout
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x


class TransformerLayer(nn.Module):
    """Multi-functional Transformer Layer of the Vanilla Transformer.

    - EncoderLayer = TransformerLayer(causal=False, receives_context=False)
    - DecoderLayer = TransformerLayer(causal=True, receives_context=True)
    - DecoderOnlyLayer = TransformerLayer(causal=True, receives_context=False)
    """

    def __init__(self, hid_dim=512, n_heads=8, pf_dim=2048, dropout=0.,
                 causal=False, receives_context=False):
        """Initializes TransformerLayer.

        Args:
            hid_dim (int): hidden dimension size
            n_heads (int): number of attention heads
            pf_dim (int): dimensions of the position-wise layer
            dropout (float): attention and position-wise layer dropouts
            causal (bool): whether to apply causal mask or not
            receives_context (float): whether layer has encoder context attention
        """

        super().__init__()

        self.causal = causal
        self.receives_context = receives_context

        # masked self-attention + layer normalization
        self.self_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, causal=causal)
        self.self_attn_ln = nn.LayerNorm(hid_dim)

        # encoder-attention + layer normalization
        if self.receives_context:
            self.enc_attn = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
            self.enc_attn_ln = nn.LayerNorm(hid_dim)

        # position-wise feedforward layer + layer normalization
        self.pf = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.pf_ln = nn.LayerNorm(hid_dim)

        # attention and position-wise feedforward layer dropouts
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask=None, context=None, context_mask=None):
        """
        Args:
            x (torch.Tensor): input sequences of shape `[bs, seq_len, hid_dim]`
            input_mask (torch.Tensor): boolean input mask of shape `[bs, seq_len]`
            context (torch.Tensor): encoder context of shape `[bs, enc_seq_len, hid_dim]`
            context_mask  (torch.Tensor): boolean encoder context mask of shape `[bs, enc_seq_len]`

        Returns:
            torch.Tensor: processed sequences of shape `[bs, seq_len, hid_dim]`
        """
        ### block 1
        # self-attention + dropout
        attn_out = self.self_attn(query=x, key=x, value=x, query_mask=input_mask)
        attn_out = self.dropout(attn_out)

        # residual (attention) + attention layer norm
        x = self.self_attn_ln(x + attn_out)

        ### block 2
        if self.receives_context and context is not None:
            # encoder-attention + dropout
            enc_attn_out = self.enc_attn(
                query=x, key=context, value=context,
                query_mask=input_mask, key_mask=context_mask
            )
            enc_attn_out = self.dropout(enc_attn_out)

            # residual (attention) + attention layer norm
            x = self.enc_attn_ln(x + enc_attn_out)

        ### block 3
        # positionwise feedforward + dropout
        ff_out = self.dropout(self.pf(x))

        # residual (positionwise feedforward) + positionwise feedforward layer norm
        x = self.pf_ln(x + ff_out)

        return x


class TransformerEncoder(nn.Module):
    """Multi-layer Transformer Encoder.

    Follows the architecture of Vanilla Transformer Encoder
    from "Attention Is All You Need".

    Modifications:
        - Learned positional embeddings instead of the sinusoidal positional encoding.
    """

    def __init__(self, num_tokens, hid_dim=512, n_layers=6, n_heads=8,
                 pf_dim=2048, dropout=0., max_len=128):
        """Initializes TransformerEncoder.

        Args:
            num_tokens (int): number of tokens in input sequences
            hid_dim (int): hidden dimension size
            n_layers (int): number of Encoder layers
            n_heads (int): number of attention heads
            pf_dim (int): dimensions of the position-wise layer
            dropout (float): attention and position-wise layer dropouts
            max_len (int): maximum lengths of input sequences.
        """

        super().__init__()

        # embeddings
        self.tok_embedding = nn.Embedding(num_tokens, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        self.dropout = nn.Dropout(dropout)

        # encoder layers (implemented below)
        self.layers = nn.ModuleList([
            TransformerLayer(
                hid_dim, n_heads, pf_dim, dropout,
                causal=False, receives_context=False
            )
            for _ in range(n_layers)
        ])

        # scale parameter
        self.scale = torch.nn.Parameter(
            torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32)),
            requires_grad=False
        )

        # custom weight initialization
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x, input_mask=None):
        """
        Args:
            x (torch.Tensor): token sequences of shape `[bs, seq_len]`
            input_mask (torch.Tensor): boolean input mask of shape `[bs, seq_len]`

        Returns:
            torch.Tensor: encoded sequences of shape `[bs, seq_len, hid_dim]`
        """
        bs, seq_len = x.shape[:2]
        device = x.device

        # get token embeddings and scale with self.scale parameter
        tok_emb = self.tok_embedding(x) / self.scale

        # get pos embeddings
        indices = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(indices)

        # sum up token and positional embeddings and apply dropout
        emb = tok_emb + pos_emb
        x = self.dropout(emb)

        # apply encoder layers one by one; input shape is [bs, seq_len, hid_dim]
        input_mask = ~input_mask.unsqueeze(-1) if input_mask is not None else None
        for layer in self.layers:
            x = layer(x, query_mask=input_mask)

        return x


class TransformerDecoder(nn.Module):
    """Multi-layer Transformer Decoder.

    Follows the architecture of Vanilla Transformer Decoder from "Attention Is All You Need".

    Outputs scores for tokens in the target sequence.

    Modifications:
        - Learned positional embeddings instead of the sinusoidal positional encoding.
        - Allows passing as input image embedding vector which is prepended to
        the token embeddings.
    """

    def __init__(self, num_tokens, hid_dim=512, n_layers=6, n_heads=8,
                 pf_dim=2048, dropout=0., max_len=128, receives_context=True):
        """Initializes TransformerDecoder.

        Args:
            num_tokens (int): number of tokens in input sequences
            hid_dim (int): hidden dimension size
            n_layers (int): number of Decoder layers
            n_heads (int): number of attention heads
            pf_dim (int): dimensions of the position-wise layer
            dropout (float): attention and position-wise layer dropouts
            max_len (int): maximum lengths of input sequences
            receives_context (float): whether layer has encoder context attention
        """

        super().__init__()

        # embeddings
        self.tok_embedding = nn.Embedding(num_tokens, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        self.dropout = nn.Dropout(dropout)

        # decoder layers (implemented below)
        self.receives_context = receives_context
        self.layers = nn.ModuleList([
            TransformerLayer(
                hid_dim, n_heads, pf_dim, dropout,
                causal=True, receives_context=receives_context
            )
            for _ in range(n_layers)
        ])

        # scale parameter
        self.scale = torch.nn.Parameter(
            torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32)),
            requires_grad=False
        )

        # output layer
        self.classifier = nn.Linear(hid_dim, num_tokens)

    def forward(self, x, input_mask=None, context=None, context_mask=None, start_emb=None):
        """
        Args:
            x (torch.Tensor): token sequences of shape `[bs, seq_len]`
            input_mask (torch.Tensor): boolean input mask of shape `[bs, seq_len]`
            context (torch.Tensor): encoder context of shape `[bs, enc_seq_len, hid_dim]`
            context_mask (torch.Tensor): boolean context mask of shape `[bs, enc_seq_len]`
            start_emb (torch.Tensor, optional): starting position embedding of shape `[bs, hid_dim]`

        Returns:
            torch.Tensor: decoded sequences of shape `[bs, seq_len, num_tokens]`
        """
        bs, dec_seq_len = x.shape[:2]
        device = x.device

        # get token embeddings
        tok_emb = self.tok_embedding(x)

        # add image embedding
        if start_emb is not None:
            tok_emb = torch.cat((start_emb.unsqueeze(1), tok_emb), 1)
            dec_seq_len += 1

        # scale token embeddings with self.scale parameter
        tok_emb = tok_emb / self.scale

        # get pos embeddings
        indices = torch.arange(dec_seq_len, device=device)
        pos_emb = self.pos_embedding(indices).unsqueeze(0)

        # sum up token and positional embeddings and apply dropout
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # apply decoder layers one by one; input shape is [bs, seq_len, hid dim]
        for layer in self.layers:
            x = layer(x, input_mask=input_mask, context=context, context_mask=context_mask)

        out = self.classifier(x)

        return out

    @torch.no_grad()
    def generate(self, start_emb, caption=None, context=None, max_len=25,
                 temperature=1.0, beam_size=10, top_k=50, eos_index=3):
        """Generates text tokens based on the image embedding.

        Args:
            start_emb (torch.Tensor): starting position embedding of shape `[1, hid_dim]`
            caption (torch.Tensor, optional): beginning tokens of the caption of shape `[1, seq_len]`
            context (torch.Tensor): encoder context of shape `[1, seq_len, hid_dim]`
            max_len (int): maximum length of the caption
            temperature (float): temperature for softmax over logits
            beam_size (int): number of maintained branches at each step
            top_k (int): number of the most probable tokens to consider during sampling
            eos_index (int): index of the EOS (end-of-sequence) token

        Returns:
            torch.Tensor: generated caption tokens of shape `[1, min(output_len, max_len)]`
        """

        assert start_emb is not None or caption is not None, \
            "Either `start_emb` or `caption` should be provided to `model.generate()`"

        device = start_emb.device

        # beam search sampling helper
        helper = BeamSearchHelper(
            temperature=temperature, beam_size=beam_size,
            top_k=top_k, eos_index=eos_index,
            device=device
        )

        sample_seq = torch.zeros((1, max_len), dtype=torch.long, device=device)
        # process caption tokens if present
        if caption is None:
            pos = 0
        else:
            pos = caption.size(1)
            sample_seq[:, :pos] = caption

        # run TransformerDecoder over the inputs and predict the next token
        outputs = self(x=sample_seq[:, :pos], context=context, start_emb=start_emb)
        logits = outputs[:, -1, :]

        # filter `top_k` values
        logits = helper.filter_top_k(logits)

        # compute probabilities and sample k values
        sample_ind = helper.sample_k_indices(logits, k=beam_size)
        sample_val = helper.filter_by_indices(logits, sample_ind).log_softmax(-1)
        sample_ind, sample_val = sample_ind.T, sample_val.T

        # update total prediction sequences
        sample_seq = sample_seq.repeat(beam_size, 1)
        sample_seq[:, pos:pos + 1] = sample_ind

        # repeat `image_emb` and `context`
        if context is not None:
            context = context.expand(beam_size, -1, -1)
        start_emb = start_emb.expand(beam_size, -1)

        for i in range(pos + 1, max_len):
            # predict the next time step
            outputs = self(x=sample_seq[:, :i], context=context, start_emb=start_emb)
            logits = outputs[:, -1, :]

            (prev_seqs, prev_vals), (new_ind, new_val) = helper.process_logits(
                logits, sample_seq, sample_val
            )

            # create candidate sequences and compute their probabilities
            prev_seqs[:, i:i + 1] = new_ind.unsqueeze(0).T
            cand_seq = prev_seqs
            cand_val = prev_vals.flatten() + new_val

            # sample `beam` sequences
            filter_ind = helper.sample_k_indices(cand_val, k=beam_size)

            # update total sequences and their scores
            sample_val = cand_val[filter_ind]
            sample_seq = cand_seq[filter_ind]

            # filter `has_ended` flags
            helper.has_ended = helper.has_ended[filter_ind]

            # check if every branch has ended
            if helper.all_ended():
                break

        # sample output sequence
        ind = helper.sample_k_indices(sample_val, k=1)
        output_seq = sample_seq[ind, :i].squeeze()

        return output_seq
