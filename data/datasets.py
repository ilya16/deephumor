import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from data import SPECIAL_TOKENS, WordPunctTokenizer


class MemeDataset(Dataset):
    """MemeGenerator dataset class."""

    def __init__(self, root, vocab, tokenizer=WordPunctTokenizer(),
                 split='train', num_classes=300, image_transform=None,
                 preload_images=True):
        assert split in ('train', 'val', 'test'), 'Incorrect data split'

        self.root = root
        self.split = split
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.image_transform = image_transform
        self.preload_images = preload_images

        self.num_classes = num_classes
        self._load_dataset()

    def _load_dataset(self):
        # load templates information
        fn_temp = os.path.join(self.root, 'templates.txt')
        assert os.path.exists(fn_temp), \
            f'Templates file {fn_temp} is not found'

        dir_imgs = os.path.join(self.root, 'images')
        assert os.path.isdir(dir_imgs), \
            f'Images directory {dir_imgs} is not found'

        self.templates = {}
        self.images = {}
        with open(fn_temp, 'r') as f:
            for line in f:
                label, _, url = line.strip().split('\t')
                filename = url.split('/')[-1]
                self.templates[label] = os.path.join(dir_imgs, filename)

                # preaload images and apply transforms
                if self.preload_images:
                    img = Image.open(self.templates[label])
                    if self.image_transform is not None:
                        img = self.image_transform(img)
                    self.images[label] = img
                else:
                    self.images[label] = self.templates[label]

                if len(self.templates) == self.num_classes:
                    break

        # load captions
        fn_capt = os.path.join(self.root, f'captions_{self.split}.txt')
        assert os.path.exists(fn_capt), \
            f'Captions file {fn_capt} is not found'

        self.captions = []
        with open(fn_capt, 'r') as f:
            for i, line in enumerate(f):
                label, _, caption = line.strip().split('\t')
                if label in self.templates:
                    self.captions.append((label, caption))

    def _preprocess_text(self, text):
        # tokenize
        tokens = self.tokenizer.tokenize(text.lower())

        # replace with `UNK`
        tokens = [tok if tok in self.vocab.stoi else SPECIAL_TOKENS['UNK'] for tok in tokens]

        # convert to ids
        tokens = [self.vocab.stoi[tok] for tok in tokens]

        return tokens

    def __getitem__(self, idx):
        label, caption = self.captions[idx]
        img = self.images[label]

        # label and caption tokens
        label = torch.tensor(self._preprocess_text(label)).long()
        caption = torch.tensor(self._preprocess_text(caption)).long()

        # image transform
        if not self.preload_images:
            img = Image.open(img)
            if self.image_transform is not None:
                img = self.image_transform(img)

        return label, caption, img

    def __len__(self):
        return len(self.captions)
