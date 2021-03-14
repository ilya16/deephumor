import torch


class BeamSearchHelper:
    """Helper class with common functions for beam search sampling."""

    def __init__(self, temperature=1.0, beam_size=10, top_k=50,
                 unk_index=1, eos_index=3, device='cuda'):
        assert beam_size <= top_k, '`beam_size` should be less than `top_k`'

        self.temperature = temperature
        self.beam_size = beam_size
        self.top_k = top_k
        self.unk_index = unk_index
        self.eos_index = eos_index
        self.device = device
        self._build_has_ended_variables()

    def _build_has_ended_variables(self):
        """Returns flags and masks for monitoring if generation has ended."""
        # flags showing if sequence has ended
        self.has_ended = torch.zeros(self.beam_size, dtype=torch.bool, device=self.device)

        # masks for filtering out predictions for ended/not_ended sequences
        self._n_copies_has_ended = torch.tensor([[self.beam_size], [1]], device=self.device)

        self._mask_has_ended = torch.stack([
            torch.ones(self.beam_size, dtype=torch.bool, device=self.device),
            torch.cat([
                torch.ones(1, dtype=torch.bool, device=self.device),
                torch.zeros(self.beam_size - 1, dtype=torch.bool, device=self.device),
            ])],
            dim=0
        )

    def filter_top_k(self, logits):
        """Filters `top_k` logit values by zeroing out others."""
        filter_ind = logits < torch.topk(logits, self.top_k, dim=-1).values[:, -1].unsqueeze(-1)
        filter_ind[:, self.unk_index] = True  # zero out unk token
        logits[filter_ind] = float('-inf')
        return logits

    def sample_k_indices(self, logits, k=None):
        """Samples `beam_size` indices for each sequence in the batch."""
        # compute probabilities
        p_next = torch.softmax(logits / self.temperature, dim=-1)

        # sample values
        k = self.beam_size if k is None else k
        sample_ind = torch.multinomial(p_next, k)

        return sample_ind

    @staticmethod
    def filter_by_indices(values, indices):
        sample_val = torch.gather(values, 1, indices)
        return sample_val

    def process_logits(self, logits, sample_seq, sample_val):
        """Main logic of beam search sampling step.

        Steps:
            - filter `top_k` logit scores
            - filter out predictions for already ended sequences
            - check if new predictions end sequence
            - update `has_ended` indices

        Args:
            logits (torch.Tensor): logit predictions, outputs of the classifier layer
            sample_seq (torch.Tensor): `beam_size` sequences from the previous sampling step
            sample_val (torch.Tensor): scores for the sequences from the previous sampling step

        Returns:
            (prev_seqs, prev_vals), (new_ind, new_val):
                expanded sequences and their scores from the previous sampling step
                + new candidate predictions and their scores
        """
        # filter `top_k` values
        logits = self.filter_top_k(logits)

        # sample `beam` sequences for each branch
        new_ind = self.sample_k_indices(logits, k=self.beam_size)
        new_val = self.filter_by_indices(logits, new_ind).log_softmax(-1)
        new_ind, new_val = new_ind.flatten(), new_val.flatten()

        # numbers of repeat_interleave copies (if ended, only a single copy)
        n_copies = self._n_copies_has_ended[self.has_ended.long(), :].flatten()

        # mask for unique rows
        unique_rows = self._mask_has_ended[self.has_ended.long(), :].flatten()

        # filter values
        new_ind = new_ind[unique_rows]
        new_val = new_val[unique_rows]

        # check if the sequences already ended
        # (no need to predict and evaluate new scores)
        self.has_ended = torch.repeat_interleave(self.has_ended, n_copies, dim=0)
        new_ind[self.has_ended], new_val[self.has_ended] = 0, 0.

        # update `had_ended` based on new predictions
        self.has_ended = self.has_ended | (new_ind == self.eos_index)

        # repeat current sampled sequences
        prev_seqs = torch.repeat_interleave(sample_seq, n_copies, dim=0)
        prev_vals = torch.repeat_interleave(sample_val, n_copies, dim=0)

        if len(prev_seqs.size()) == 1:
            prev_seqs = prev_seqs.unsqueeze(0)
            prev_vals = prev_vals.unsqueeze(0)

        return (prev_seqs, prev_vals), (new_ind, new_val)

    def all_ended(self):
        """Returns bool indicating if all sequences have ended."""
        return torch.all(self.has_ended)
