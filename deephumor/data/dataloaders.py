import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    """Batch collate with padding for Dataloader."""
    # unpack batch
    labels, captions, images = zip(*batch)

    # pad sequences
    lengths = torch.tensor(list(map(len, captions)))
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    images = torch.stack(images, dim=0)

    return labels, captions, images, lengths
