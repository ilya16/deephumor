"""Vocabulary tools."""

from collections import Counter

SPECIAL_TOKENS = {
    'PAD': '<pad>',
    'UNK': '<unk>',
    'BOS': '<bos>',
    'EOS': '<eos>',
    'SEP': '<sep>',
    'EMPTY': '<emp>',
}


class Vocab:
    """Token vocabulary."""

    def __init__(self, tokens, special_tokens=SPECIAL_TOKENS.values()):
        tokens = list(sorted(filter(lambda x: x not in special_tokens, tokens)))
        self.tokens = list(special_tokens) + tokens
        self.stoi = {self.tokens[idx]: idx for idx in range(len(self.tokens))}
        self.itos = {idx: self.tokens[idx] for idx in range(len(self.tokens))}

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def save(self, filepath):
        with open(filepath, 'w') as f:
            for token in self.tokens:
                f.write(f'{token}\t{self.stoi[token]}\n')

    @staticmethod
    def load(filepath):
        tokens = []
        with open(filepath, 'r') as f:
            for line in f:
                token, _ = line.strip().split('\t')
                tokens.append(token)
        return Vocab(tokens)


def build_vocab(documents, tokenizer, min_df=7):
    """Builds vocabulary of tokens from a collection of documents.

    Args:
        documents (list[str]): collection of documents
        tokenizer (Tokenizer): Tokenizer object
        min_df (int): minimum document frequency for tokens

    Returns:
        Vocab: vocabulary of tokens
    """
    token_counts = Counter()

    # tokenize and count unique tokens
    for text in documents:
        tokens = set(tokenizer.tokenize(text.lower()))
        token_counts.update(tokens)

    # filter by minimum document frequency
    tokens = [token for token, count in token_counts.items() if count >= min_df]

    # build vocabulary
    vocab = Vocab(tokens)

    return vocab


def build_vocab_from_file(captions_file, tokenizer, min_df=7):
    """Builds vocabulary from captions file.

    Args:
        captions_file (str): path to the file with captions
        tokenizer (Tokenizer): Tokenizer object
        min_df (int): minimum document frequency for tokens

    Returns:
        Vocab: vocabulary of tokens
    """

    captions = []
    with open(captions_file) as f:
        for line in f:
            _, _, caption = line.strip().split('\t')
            captions.append(caption)

    return build_vocab(captions, tokenizer, min_df=min_df)
