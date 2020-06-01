"""RNN-based models."""
import torch
from torch import nn

from deephumor.models.beam import BeamSearchHelper


class LSTMDecoder(nn.Module):
    """LSTM-based decoder."""

    def __init__(self, num_tokens, emb_dim=256, hidden_size=512,
                 num_layers=3, dropout=0.1, embedding=None):

        super(LSTMDecoder, self).__init__()

        self.num_tokens = num_tokens
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(num_tokens, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True,
                            dropout=(0 if num_layers == 1 else dropout))

        self.classifier = nn.Linear(hidden_size, num_tokens)

    def forward(self, image_emb, captions, lengths=None):
        # caption tokens embeddings
        token_emb = self.embedding(captions)

        # image embedding + token embeddings
        x = torch.cat((image_emb.unsqueeze(1), token_emb), dim=1)

        if lengths is None:
            lengths = torch.tensor(x.size(1)).repeat(x.size(0))

        # LSTM ouputs
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # mapping into `num_tokens`
        outputs = self.classifier(outputs)

        return outputs

    def generate(self, image_emb, caption=None, max_len=25,
                 temperature=1.0, beam_size=10, top_k=50, eos_index=3):
        """Generates text tokens based on the image embedding.

        Args:
            image_emb (torch.Tensor): image embedding of shape `[1, emb_dim]`
            caption (torch.Tensor, optional): beginning tokens of the caption of shape `[1, seq_len]`
            max_len (int): maximum length of the caption
            temperature (float): temperature for softmax over logits
            beam_size (int): number of maintained branches at each step
            top_k (int): number of the most probable tokens to consider during sampling
            eos_index (int): index of the EOS (end-of-sequence) token

        Returns:
            torch.Tensor: generated caption tokens of shape `[1, min(output_len, max_len)]`
        """

        # beam search sampling helper
        helper = BeamSearchHelper(
            temperature=temperature, beam_size=beam_size,
            top_k=top_k, eos_index=eos_index,
            device=image_emb.device
        )

        # process caption tokens if present
        if caption is None:
            inputs = image_emb
        else:
            token_emb = self.embedding(caption)
            inputs = torch.cat([image_emb, token_emb], dim=1)

        # run LSTM over the inputs and predict the next token
        outputs, (h, c) = self.lstm(inputs)
        logits = self.classifier(outputs[:, -1, :])

        # repeat hidden state `beam` times
        h, c = h.repeat((1, beam_size, 1)), c.repeat((1, beam_size, 1))

        # filter `top_k` values
        logits = helper.filter_top_k(logits)

        # compute probabilities and sample k values
        sample_ind = helper.sample_k_indices(logits, k=beam_size)
        sample_val = helper.filter_by_indices(logits, sample_ind).log_softmax(-1)
        sample_ind, sample_val = sample_ind.T, sample_val.T

        # define total prediction sequences
        sample_seq = sample_ind.clone().detach()
        if caption is not None:
            sample_seq = torch.cat([caption.repeat(beam_size, 1), sample_seq], dim=1)

        # reusable parameters
        beam_copies = torch.tensor([beam_size] * beam_size).to(outputs.device)

        # update `has_ended` index
        helper.has_ended = (sample_ind == eos_index).view(-1)

        for i in range(sample_seq.size(1), max_len):
            # predict the next time step
            inputs = self.embedding(sample_ind)
            outputs, (h, c) = self.lstm(inputs, (h, c))
            logits = self.classifier(outputs[:, -1, :])

            (prev_seqs, prev_vals), (new_ind, new_val) = helper.process_logits(
                logits, sample_seq, sample_val
            )

            # create candidate sequences and compute their probabilities
            cand_seq = torch.cat((prev_seqs, new_ind.unsqueeze(0).T), -1)
            cand_val = prev_vals.flatten() + new_val

            # sample `beam` sequences
            filter_ind = helper.sample_k_indices(cand_val, k=beam_size)

            # update total sequences and their scores
            sample_val = cand_val[filter_ind]
            sample_seq = cand_seq[filter_ind]
            sample_ind = sample_seq[:, -1].unsqueeze(-1)

            # filter `has_ended` flags
            helper.has_ended = helper.has_ended[filter_ind]

            # check if every branch has ended
            if helper.all_ended():
                break

            # repeat hidden state `beam` times and filter by sampled indices
            h = torch.repeat_interleave(h, beam_copies, dim=1)
            c = torch.repeat_interleave(c, beam_copies, dim=1)
            h, c = h[:, filter_ind, :], c[:, filter_ind, :]

        # sample output sequence
        ind = helper.sample_k_indices(sample_val, k=1)
        output_seq = sample_seq[ind, :].squeeze()

        return output_seq
