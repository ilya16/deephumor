"""Text Tokenizers."""
import abc
import re


class Tokenizer:
    """Abstract tokenizer."""

    @abc.abstractmethod
    def tokenize(self, text):
        pass


class WordPunctTokenizer:
    """WordPunctuation tokenizer."""

    token_pattern = re.compile(r"[<\w'>]+|[^\w\s]+")

    def tokenize(self, text):
        return self.token_pattern.findall(text)


class CharTokenizer:
    """Character-level tokenizer that preserves special tokens in `<>`."""

    token_pattern = re.compile(r"<\w+>|.")

    def tokenize(self, text):
        return self.token_pattern.findall(text)
