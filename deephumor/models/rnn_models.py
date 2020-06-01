"""RNN-based models."""
import torch
from torch import nn


class LSTMDecoder(nn.Module):
    """LSTM-based decoder."""

    def __init__(self, num_tokens, emb_dim=256, hidden_size=512, num_layers=3, dropout=0.1,
                 embedding=None):

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
        filter_ind = logits < torch.topk(logits, top_k).values[0, -1]  # [..., -1, None]
        filter_ind[:, 1] = True  # zero out unk token
        logits[filter_ind] = float('-inf')

        # compute probabilities and sample k values
        p_next = torch.softmax(logits / temperature, dim=-1)
        sample_ind = torch.multinomial(p_next, beam_size).transpose(0, 1)
        sample_val = logits.log_softmax(-1).squeeze()[sample_ind]

        # define total prediction sequences
        sample_seq = sample_ind.clone().detach()
        if caption is not None:
            sample_seq = torch.cat([caption.repeat(beam_size, 1), sample_seq], dim=1)

        # reusable parameters
        beam_copies = torch.tensor([beam_size] * beam_size).to(outputs.device)

        # flags showing if sequence has ended
        has_ended = (sample_ind == eos_index).view(-1)

        # masks for filtering out predictions for ended/not_ended sequences
        n_copies_has_ended = torch.tensor([[beam_size], [1]]).to(inputs.device)
        mask_has_ended = torch.stack(
            [torch.tensor([True] * beam_size),
             torch.tensor([True] + [False] * (beam_size - 1))],
            dim=0
        ).to(inputs.device)

        for i in range(sample_seq.size(1), max_len):
            # predict the next time step
            inputs = self.embedding(sample_ind)
            outputs, (h, c) = self.lstm(inputs, (h, c))
            logits = self.classifier(outputs[:, -1, :])

            # filter `top_k` values
            filter_ind = logits < torch.topk(logits, top_k, -1).values[:, -1].unsqueeze(-1)
            filter_ind[:, 1] = True  # zero out unk token
            logits[filter_ind] = float('-inf')

            # sample `beam` sequences for each branch
            p_next = torch.softmax(logits / temperature, dim=-1)
            new_ind = torch.multinomial(p_next, beam_size)
            new_val = torch.gather(logits, 1, new_ind).log_softmax(-1).flatten()
            new_ind = new_ind.flatten()

            # numbers of repeat_interleave copies (if ended, only a single copy)
            n_copies = n_copies_has_ended[has_ended.long(), :].flatten()

            # mask for unique rows
            unique_rows = mask_has_ended[has_ended.long(), :].flatten()

            # filter values
            new_ind = new_ind[unique_rows]
            new_val = new_val[unique_rows]

            # check if the sequences already ended
            # (no need to predict and evaluate new scores)
            has_ended = torch.repeat_interleave(has_ended, n_copies, dim=0)
            new_ind[has_ended], new_val[has_ended] = 0, 0.

            # update `had_ended` based on new predictions
            has_ended = has_ended | (new_ind == eos_index)

            # repeat current sampled sequences
            prev_seqs = torch.repeat_interleave(sample_seq.squeeze(0), n_copies, dim=0)
            prev_vals = torch.repeat_interleave(sample_val.squeeze(0), n_copies, dim=0)

            # create candidate sequencdes and compute their probabilites
            cand_seq = torch.cat((prev_seqs, new_ind.unsqueeze(0).T), -1)
            cand_val = prev_vals.flatten() + new_val
            p_next = torch.softmax(cand_val / temperature, dim=-1)

            # sample `beam` sequences
            filter_ind = torch.multinomial(p_next, beam_size)

            # update total sequences and their scores
            sample_val = cand_val[filter_ind]
            sample_seq = cand_seq[filter_ind]
            sample_ind = sample_seq[:, -1].unsqueeze(-1)

            # filter `has_ended` flags
            has_ended = has_ended[filter_ind]

            # check if every branch has ended
            if torch.all(has_ended):
                break

            # repeat hidden state `beam` times and filter by sampled indices
            h = torch.repeat_interleave(h, beam_copies, dim=1)
            c = torch.repeat_interleave(c, beam_copies, dim=1)
            h, c = h[:, filter_ind, :], c[:, filter_ind, :]

        # sample output sequence
        p = torch.softmax(sample_val / temperature, dim=-1)
        ind = torch.multinomial(p, 1)
        output_seq = sample_seq[ind, :].squeeze()

        return output_seq
