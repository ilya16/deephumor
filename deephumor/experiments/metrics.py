"""Evaluation metrics."""
from deephumor.models.utils import get_mask_from_lengths


def perplexity(logits, targets, lengths):
    log_values = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze()
    log_values /= lengths.unsqueeze(1)  # divide by lengths
    log_values[~get_mask_from_lengths(lengths)] = 0.  # remove padded indices
    pp_seq = (-log_values.sum(dim=-1)).exp()  # compute per-sequence perplexity
    return pp_seq.mean()
