import argparse
import os
from collections import defaultdict

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset split')

    parser.add_argument('--source', '-s', type=str, default='memegenerator.net',
                        help='data source')
    parser.add_argument('--data-dir', '-d', required=True, type=str,
                        help='directory with the dataset')

    parser.add_argument('--splits', type=int, default=(2500, 250, 250), nargs='3',
                        help='sizes of train/val/test splits for each template')
    parser.add_argument('--random-state', type=int, default=0,
                        help='random seed for the data shuffling')

    args = parser.parse_args()
    assert args.source == 'memegenerator.net', 'Only memegenerator.net is supported'

    np.random.seed(0)

    labels, captions = defaultdict(bool), defaultdict(list)
    with open(os.path.join(args.data_dir, 'captions.txt'), 'r') as f:
        for line in f:
            label, _, _ = line.strip().split('\t')
            captions[label].append(line)
            labels[label] = True

    f_train = open(os.path.join(args.data_dir, 'captions_train.txt'), 'w')
    f_val = open(os.path.join(args.data_dir, 'captions_val.txt'), 'w')
    f_test = open(os.path.join(args.data_dir, 'captions_test.txt'), 'w')

    for label in labels.keys():
        indices = np.arange(len(captions[label]))
        np.random.shuffle(indices)

        for idx in sorted(indices[:args.splits[0]]):
            f_train.write(captions[label][idx])

        for idx in sorted(indices[args.splits[0]:args.splits[1]]):
            f_val.write(captions[label][idx])

        for idx in sorted(indices[args.splits[1]:]):
            f_test.write(captions[label][idx])

    f_train.close()
    f_val.close()
    f_test.close()
