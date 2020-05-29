import os
import re
import time
from multiprocessing import Pool

import numpy as np
import requests
from Levenshtein import ratio as sim_ratio
from lxml import html

from .utils import time_to_str, load_image
from deephumor.data.utils import clean_text, check_text, english_prob


def crawl_templates(page=1):
    """Crawls templates from All-time page.

    Args:
        page (int): page number
    """

    meme_templates = []
    url = f'https://memegenerator.net/memes/popular/alltime/page/{page}'

    try:
        r = requests.get(url)
        tree = html.fromstring(r.content)

        divs = tree.xpath('//div[@class="char-img"]/a')

        for div in divs:
            link = div.get('href')
            img = div.find('img')
            label = img.get('alt')
            src = img.get('src')

            meme_templates.append({'label': label, 'link': link, 'src': src})
    except ConnectionError as e:
        print(e)

    return meme_templates


def crawl_template_page(template_link, page=1, num_retries=10):
    """Crawls data from the template page.

    Args:
        template_link (str): link identifier of the template
        page (int): page number
        num_retries (int): number of retries
    """

    url = f'https://memegenerator.net{template_link}/images/popular/alltime/page/{page}'
    score_pattern = re.compile(r'(-?\d+(,\d*)?)')

    num_errors = 0
    try:
        while True:
            r = requests.get(url)
            if r.status_code == 200:
                break
            else:
                num_errors += 1
                if num_errors > num_retries:
                    print('Failed to load ' + url)
                    return None, None, None
    except ConnectionError as e:
        print(e)
        return None, None, None

    tree = html.fromstring(r.content)

    label = tree.xpath('//h1/a/text()')[0]
    divs = tree.xpath('//div[@class="char-img"]')

    memes = []

    for div in divs:
        score = div.xpath('.//div[contains(@class, "score")]/text()')[0]
        score = int(score_pattern.findall(score)[0][0].replace(',', ''))
        text0 = div.xpath('a//div[@class="optimized-instance-text0"]/text()')
        text1 = div.xpath('a//div[@class="optimized-instance-text1"]/text()')
        text0 = text0[0] if text0 else ''
        text1 = text1[0] if text1 else ''

        memes.append((score, text0, text1))

    return label, memes, template_link


class MemeGeneratorCrawler:
    """MemeGenerator.net website crawler."""

    # characteristics of the website
    temp_pp = 15  # templates per page
    capt_pp = 15  # captions per page

    def __init__(self, poolsize=2,
                 min_len=10, max_len=96, max_tokens=31,
                 detect_english=False, detect_duplicates=False):
        """Initializes crawler and multiprocessing Pool.

        Args:
            poolsize (int): size of the multiprocessing pool
            min_len (int): minimum length of the caption text
            max_len (int): maximum length of the caption text
            max_tokens (int): maximum number of tokens in the caption text
            detect_english (bool): (non-stable) globally filter non-english templates
            detect_duplicates (bool): (slow) check for the similarity of captions and filter duplicates
        """

        self.poolsize = poolsize
        self.pool = Pool(poolsize)

        # text preprocessing parameters
        self.min_len = min_len
        self.max_len = max_len
        self.max_tokens = max_tokens
        self.detect_english = detect_english
        self.detect_duplicates = detect_duplicates

        # containers shared across threads
        self.captions = {}
        self.num_visited = {}
        self.total_texts = {}

    def template_page_callback(self, result):
        """Processes the results from the template page."""
        _, memes, link = result

        # check and clear memes
        memes_filtered = []

        for meme in memes:
            (score, top, bottom) = meme
            top, bottom = clean_text(top), clean_text(bottom)
            text = (top + ' ' + bottom).lower()

            if check_text(text, min_len=self.min_len, max_len=self.max_len, max_tokens=self.max_tokens):
                memes_filtered.append((score, top, bottom))
                self.total_texts[link] += text + ' '

        self.captions[link] += memes_filtered
        self.num_visited[link] += 1

    def crawl_dataset(self, num_templates=300, num_captions=3000, save_dir='memes'):
        """Crawls dataset from memegenerator.net website.

        Args:
            num_templates (int): number of meme templates to crawl
            num_captions (int): number of captions per template
            save_dir (str): directory for saving the data
        """
        # approximate number of caption pages needed
        num_capt_pages = int(num_captions / self.capt_pp)
        num_capt_pages += (10 - num_capt_pages % 10)

        # directories and files
        images_dir = os.path.join(save_dir, "images/")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        templates_file = open(os.path.join(save_dir, "templates.txt"), 'a')
        captions_file = open(os.path.join(save_dir, "captions.txt"), 'a')

        # counters
        temp_page = 1
        total_captions, total_templates = 0, 0

        # start crawling until enough templates are loaded
        start_time = time.time()
        while total_templates < num_templates:
            # parse page with templates
            templates = crawl_templates(page=temp_page)
            print(f'{time_to_str(time.time() - start_time)}, '
                  f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                  f'Crawling page {temp_page} with {len(templates)} templates')

            # load captions in async mode
            for temp in templates:
                link = temp['link']
                self.captions[link] = []
                self.num_visited[link] = 0
                self.total_texts[link] = ''

                for i in range(1, num_capt_pages + 1):
                    self.pool.apply_async(crawl_template_page, [link, i],
                                          callback=self.template_page_callback)
                time.sleep(0.3)

            total_page_templates, total_page_captions = 0, 0
            for temp in templates:
                label, link, src = temp['label'], temp['link'], temp['src']

                # wait until all initial pages for the template are loaded
                for n_retry in range(100):
                    if self.num_visited[link] >= num_capt_pages:
                        break
                    time.sleep(0.5)

                if self.detect_english:
                    # check captions language
                    prob_en = np.mean([english_prob(self.total_texts[link]) for _ in range(5)])
                    if prob_en < 0.9:
                        # non-english, stop processing
                        print(f'{time_to_str(time.time() - start_time)}, '
                              f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                              f'   NON_ENGLISH {label} - {len(self.captions[link])} captions (eng:{prob_en:.3f})')
                        continue
                else:
                    prob_en = None

                page = num_capt_pages
                if self.detect_duplicates:
                    # check duplicates and keep collecting to get `n_captions_per_template`

                    unique_captions = []
                    while True:
                        for n_retry in range(100):
                            if self.num_visited[link] >= page:
                                break
                            time.sleep(0.5)

                        if not self.captions[link]:
                            # no new captions for the template
                            break

                        # process crawled captions for duplicates (slow..)
                        for (score, top, bottom) in self.captions[link]:
                            is_unique = True
                            text = (top + ' ' + bottom).lower()

                            for (_, other_top, other_bottom) in unique_captions:
                                other_text = (other_top + ' ' + other_bottom).lower()
                                if sim_ratio(text, other_text) > 0.9:
                                    is_unique = False
                                    break

                            if is_unique:
                                unique_captions.append((score, top, bottom))

                        self.captions[link] = []
                        if len(unique_captions) >= num_captions:
                            break

                        # load five more pages
                        for i in range(page + 1, page + 10):
                            self.pool.apply_async(crawl_template_page, [link, i],
                                                  callback=self.template_page_callback)
                        page = i
                else:
                    unique_captions = self.captions[link]

                # total captions
                if len(unique_captions) < num_captions:
                    # skip template
                    print(f'{time_to_str(time.time() - start_time)}, '
                          f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                          f'   NOT_ENOUGH {label} - {len(unique_captions)} captions (eng:{prob_en:.3f})')
                    continue

                # take top captions by their score
                captions = list(sorted(unique_captions, key=lambda x: -x[0]))
                captions = captions[:num_captions]

                # save template information and load image
                templates_file.write(f'{label}\t{link}\t{src}\n')
                self.pool.apply_async(load_image, [src, images_dir])
                total_templates += 1
                total_page_templates += 1

                # save captions
                for (score, top, bot) in captions:
                    top = top if top else '<et>'
                    bot = bot if bot else '<eb>'
                    text = top + ' <sep> ' + bot
                    captions_file.write(f'{label}\t{score}\t{text}\n')

                total_captions += len(captions)
                total_page_captions += len(captions)

                # delete data from memory
                del self.captions[link]
                del self.num_visited[link]
                del self.total_texts[link]

                print(f'{time_to_str(time.time() - start_time)}, '
                      f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                      f'   {label} - {len(captions)} captions ({total_captions}) (pid:{page}, en:{prob_en:.3f})')

                if total_templates == num_templates:
                    # crawled enough templates, skip others if any
                    break

            print(f'{time_to_str(time.time() - start_time)}, '
                  f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                  f'Crawled  page {temp_page} with {total_page_templates} templates '
                  f'and {total_page_captions} captions ({total_templates}/{total_captions})')

            time.sleep(0.5)
            temp_page += 1

        print(f'{time_to_str(time.time() - start_time)}: '
              f'Finished: crawled {total_templates} templates and '
              f'{total_captions} captions')

        templates_file.close()
        captions_file.close()
