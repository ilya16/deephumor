import numpy as np
from PIL import ImageFont, ImageDraw
from copy import deepcopy


MEME_FONT_PATH = '../../fonts/impact.ttf'


def memeify_image(img, top='', bottom='', font_path=MEME_FONT_PATH):
    """Adds top and bottom captions to an image.

    Args:
        img (PIL.Image): input image
        top (str): top caption text
        bottom (str): top caption text
        font_path (str): path to font

    Returns:
        PIL.Image: captioned image
    """
    # do not change existing image
    img = deepcopy(img)

    # initial font
    font = _get_initial_font(img, texts=[top, bottom], font_path=font_path)

    # split texts into lines
    top_lines = split_to_lines(img, top, font)
    bottom_lines = split_to_lines(img, bottom, font)

    # adjust the font
    font = _get_final_font(img, [top_lines, bottom_lines], font_path=font_path)

    # caption image with both texts
    img = caption_image(img, top_lines, font, 'top')
    img = caption_image(img, bottom_lines, font, 'bottom')

    return img


def get_maximal_font(img, text, font_size=64, text_width=0.94, font_path=MEME_FONT_PATH):
    """Computes the font of maximal size that fits the text.

    Args:
        img (PIL.Image): input image
        text (str): text to fit into image
        font_size (int): initial font size
        text_width (float): text width ratio with respect to image width
        font_path (str): path to font

    Returns:
        PIL.ImageFont: optimal font
    """
    font = ImageFont.truetype(font_path, font_size)
    w, h = font.getsize(text)

    # find the biggest font size that works
    while w > img.width * text_width:
        font_size = font_size - 1
        font = ImageFont.truetype(font_path, font_size)
        w, h = font.getsize(text)

    return font


def _get_initial_font(img, texts, max_chars=20, font_path=MEME_FONT_PATH):
    """Compute initial font of maximal size based of texts.

    Args:
        img (PIL.Image): input image
        texts (List[str]): list of texts
        max_chars (int): maximum number of characters in a line
        font_path (str): path to font

    Returns:
        PIL.ImageFont: optimal font
    """
    # compute the maximum number of characters in a line
    max_len = max(map(len, texts))
    max_len = max_len if max_len < max_chars else max_chars
    longest_text = 'G' * max_len

    # get initial font size from image dimensions
    font_size = int(img.height / 5.4)

    # get maximal font for the initial text
    font = get_maximal_font(img, longest_text, font_size, font_path=font_path)

    return font


def _get_final_font(img, text_lines, font_path=MEME_FONT_PATH):
    """Compute final font of maximal size based of texts split into lines.

    Args:
        img (PIL.Image): input image
        text_lines (List[List[str]]): list of list of text lines
        font_path (str): path to font

    Returns:
        PIL.ImageFont: optimal font
    """
    # initial font size
    font_size = int(img.height / 5.4) // max(map(len, text_lines))
    font = ImageFont.truetype(font_path, font_size)

    # find the text with the highest occupied width
    text_lines = [text for lines in text_lines for text in lines]
    lengths = list(map(lambda x: font.getsize(x)[0], text_lines))
    longest_text = text_lines[np.argmax(lengths)]

    # get maximal font for the text with highest width
    font = get_maximal_font(img, longest_text, font_size, font_path=font_path)

    return font


def split_to_lines(img, text, font):
    """Splits text into lines to fit the image with a given font.

    Args:
        img (PIL.Image): input image
        text (str): input text
        font (PIL.ImageFont): text font

    Returns:
        List[str]: list of text lines
    """
    draw = ImageDraw.Draw(img)
    text = text.replace('', '').upper()
    w, h = draw.textsize(text, font)  # measure the size the text will take

    # compute the number of lines
    line_count = 1
    if w > img.width:
        line_count = w // img.width + 1

    lines = []
    if line_count > 1:
        # cut text into lines preserving words

        last_cut = 0
        is_last = False

        for i in range(0, line_count):
            cut = (len(text) // line_count) * i if last_cut == 0 else last_cut

            if i < line_count - 1:
                next_cut = (len(text) // line_count) * (i + 1)
            else:
                next_cut = len(text)
                is_last = True

            # make sure we don't cut words in half
            if not (next_cut == len(text) or text[next_cut] == " "):
                while text[next_cut] != " ":
                    next_cut += 1

            line = text[cut:next_cut].strip()

            # does line still fit?
            w, h = draw.textsize(line, font)
            if not is_last and w > img.width * 0.95:
                next_cut -= 1
                while text[next_cut] != " ":
                    next_cut -= 1

            last_cut = next_cut
            lines.append(text[cut:next_cut].strip())
    else:
        lines.append(text)

    return lines


def caption_image(img, text_lines, font, pos='top'):
    """Captions the image with text.

    Args:
        img (PIL.Image): input image
        text_lines (List[str]): list of text lines
        font (PIL.ImageFont): text font
        pos (str): position of text (`top` or `bottom`)

    Returns:
        PIL.Image: captioned image
    """
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(text_lines[0], font)  # measure the size the text will take

    # text border size
    border_size = font.size // 18

    # compute the position of text on y-axis
    last_y = -h
    if pos == 'bottom':
        last_y = img.height * 0.987 - h * (len(text_lines) + 1) - border_size

    # draw text lines
    for line in text_lines:
        w, h = draw.textsize(line, font)
        x = img.width / 2 - w / 2
        y = last_y + h

        # add borders of black color
        for xx in range(-border_size, border_size + 1):
            for yy in range(-border_size, border_size + 1):
                draw.text((x + xx, y + yy), line, (0, 0, 0), font=font)

        # add text in white
        draw.text((x, y), line, (255, 255, 255), font=font)

        last_y = y

    return img
