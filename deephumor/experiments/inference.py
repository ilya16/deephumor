import re

import torch

from deephumor.data import SPECIAL_TOKENS


PUNCT_PATTERN = re.compile(r"( )([!#$%&\()*+,\-.\/:;<=>?@\\^{|}~]+)")


def text_to_seq(text, vocab, tokenizer):
    """Transforms string text into a tensor of tokens.

    Args:
        text (str): text input
        vocab (Vocab): token vocabulary
        tokenizer (Tokenizer): text tokenizer

    Returns:
        Torch.tensor: sequence of tokens of size (1, seq_len)
    """

    # tokenize
    tokens = tokenizer.tokenize(text.lower())

    # replace with `UNK`
    tokens = [tok if tok in vocab.stoi else SPECIAL_TOKENS['UNK'] for tok in tokens]

    # convert to ids
    tokens = [vocab.stoi[tok] for tok in tokens]

    return torch.tensor(tokens).unsqueeze(0)


def seq_to_text(seq, vocab, delimiter=' '):
    """Transforms torch tensor of tokens into a text.

    Args:
        seq (Torch.tensor): sequence of tokens of size (1, seq_len)
        vocab (Vocab): token vocabulary
        delimiter (str): delimiter between text tokens

    Returns:
        str: transformed text
    """

    # find the end the sequence
    eos_ids = torch.where(seq == vocab.stoi[SPECIAL_TOKENS['EOS']])[0]
    if len(eos_ids) > 0:
        seq = seq[:eos_ids[0]]

    # convert tokens indices into text tokens
    tokens = list(map(lambda x: vocab.itos[x], seq.cpu().numpy()))

    # join text tokens
    text = delimiter.join(tokens)

    return text


def split_caption(text, num_blocks=None):
    """Splits text caption into blocks according to the special tokens.

    Args:
        text (str): input caption text
        num_blocks (int): number of blocks to return (`None` for keeping all)

    Returns:
        List[str]: a list of text blocks
    """

    def _clean_text_block(text_block):
        text_block = re.sub(r'<\w+>', '', text_block)
        text_block = re.sub(r'^\s+', '', text_block)
        text_block = re.sub(r'\s+$', '', text_block)
        text_block = PUNCT_PATTERN.sub('\\2', text_block)
        return text_block

    text_blocks = text.split(SPECIAL_TOKENS['SEP'])

    # clean blocks from any special tokens and padding spaces
    text_blocks = [_clean_text_block(t) for t in text_blocks]

    if num_blocks is None:
        num_blocks = len(text_blocks)
    elif len(text_blocks) < num_blocks:
        text_blocks += [''] * (num_blocks - len(text_blocks))

    return text_blocks[:num_blocks]
