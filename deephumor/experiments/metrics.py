"""Evaluation metrics."""


def perplexity(logits, targets, lengths, pad_index=0):
    log_values = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze()
    log_values /= lengths.unsqueeze(1)  # divide by lengths
    log_values[targets == pad_index] = 0.  # remove padded indices
    pp_seq = (-log_values.sum(dim=-1)).exp()  # compute per-sequence perplexity
    return pp_seq.mean()
