import os
import shutil

import requests


def time_to_str(time):
    """Converts time in seconds into pretty-looking string."""
    return f'{int(time / 60.):3d}:{(time % 60.):05.2f}'


def load_image(image_url, save_dir='.'):
    """Loads image by url.

    Args:
        image_url (str): image URL
        save_dir (str): directory for saving the image

    Returns:
        str: name of the file
    """
    r = requests.get(image_url, stream=True)
    file_name = image_url.split('/')[-1]
    image_path = os.path.join(save_dir, file_name)

    with open(image_path, 'wb') as out:
        shutil.copyfileobj(r.raw, out)

    return file_name
