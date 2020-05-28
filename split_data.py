import argparse
import os
from collections import defaultdict

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset split')

    parser.add_argument('--data-dir', '-d', required=True, type=str,
                        help='directory with the dataset')
    parser.add_argument('--splits', type=int, default=(2500, 250, 250), nargs=3,
                        help='sizes of train/val/test splits for each template')
    parser.add_argument('--random-state', type=int, default=0,
                        help='random seed for the data shuffling')

    args = parser.parse_args()
    assert args.source == 'memegenerator.net', 'Only memegenerator.net is supported'

    np.random.seed(0)
    start_ids = np.cumsum([0] + args.splits)
    end_ids = start_ids[1:]

    labels, captions = defaultdict(bool), defaultdict(list)
    with open(os.path.join(args.data_dir, 'captions.txt'), 'r') as f:
        for line in f:
            label, _, _ = line.strip().split('\t')
            captions[label].append(line)
            labels[label] = True

    splits = ['train', 'val', 'test']
    f_splits = [
        open(os.path.join(args.data_dir, f'captions_{split}.txt'), 'w')
        for split in splits
    ]

    for label in labels.keys():
        indices = np.arange(len(captions[label]))
        np.random.shuffle(indices)

        for i, f in enumerate(f_splits):
            for idx in sorted(indices[start_ids[i]:end_ids[i]]):
                f.write(captions[label][idx])

    for f in f_splits:
        f.close()
