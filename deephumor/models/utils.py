from inspect import isfunction

import torch


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
