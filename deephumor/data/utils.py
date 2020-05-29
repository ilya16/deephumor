import re

from langdetect import detect_langs

TOKEN_PATTERN = re.compile(r"[<\w'>]+|[!#$%&\()*+,\-./:;=?@\\^{|}~]+")
PUNCT_PATTERN_0 = re.compile(r"([<>|\\])+")
PUNCT_PATTERN_1 = re.compile(r"([%&\()*+,\-/:;=@^{}~\"])+")
PUNCT_PATTERN_23 = re.compile(r"([\.?!$#_]){4,}")


def clean_text(text):
    """Cleans text from unnecessary punctuation repetitions"""
    text = text if text else ''

    if text:
        text = PUNCT_PATTERN_0.sub('', text)
        text = PUNCT_PATTERN_1.sub(r'\g<1>', text)
        text = PUNCT_PATTERN_23.sub(r'\g<1>\g<1>\g<1>', text)

    return " ".join(text.split())


def check_text(text, min_len=10, max_len=100, max_tokens=32):
    """Checks characters and length of the text."""
    # check non-english characters
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        return False

    # filter long texts
    if len(text) < min_len or len(text) > max_len:
        return False

    # filter texts with many tokens
    if len(TOKEN_PATTERN.findall(text)) > max_tokens:
        return False

    return True


def english_prob(text):
    """Returns the probability of the text to be english text."""
    langs = detect_langs(text)
    for lang in langs:
        if lang.lang == 'en':
            return lang.prob
    return 0.
