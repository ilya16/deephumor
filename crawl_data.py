import argparse

from deephumor.crawlers import MemeGeneratorCrawler

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Meme dataset crawler')

    parser.add_argument('--source', '-s', type=str, default='memegenerator.net',
                        help='data source')
    parser.add_argument('--save-dir', '-d', required=True, type=str,
                        help='directory where the dataset should be stored')

    # crawling arguments
    parser.add_argument('--poolsize', '-p', type=int, default=25,
                        help='size of the multiprocessing Pool')
    parser.add_argument('--num-templates', '-t', type=int, default=300,
                        help='number of templates to crawl')
    parser.add_argument('--num-captions', '-c', type=int, default=1000,
                        help='number of captions per template')

    parser.add_argument('--detect-english', action='store_true',
                        help='filter out templates with majority of english texts')
    parser.add_argument('--detect-duplicates', action='store_true',
                        help='(slow) filter out duplicate captions')

    parser.add_argument('--min-len', type=int, default=10,
                        help='minimum length of the caption text')
    parser.add_argument('--max-len', type=int, default=96,
                        help='maximum length of the caption text')
    parser.add_argument('--max-tokens', type=int, default=31,
                        help='maximum number of tokens in the caption text')

    args = parser.parse_args()
    assert args.source == 'memegenerator.net', 'Only memegenerator.net is supported'

    crawler = MemeGeneratorCrawler(
        poolsize=args.poolsize,
        min_len=args.min_len, max_len=args.max_len, max_tokens=args.max_tokens,
        detect_english=args.detect_english, detect_duplicates=args.detect_duplicates
    )

    crawler.crawl_dataset(
        num_templates=args.num_templates,
        num_captions=args.num_captions,
        save_dir=args.save_dir
    )
