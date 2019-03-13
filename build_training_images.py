import glob
import cv2
import os
import time

from PIL import Image

import character_segmenter


def train_from_folder(src_directory='training_images', target_directory='letter_images'):
    start_time = time.time()
    files = glob.glob(src_directory + "/*.jpg")

    print(f'Creating letter images from {len(files)} files')

    for _, filename in enumerate(files):
        image = Image.fromarray(cv2.imread(filename))
        captcha_text = filename.split('.')[-2].split('_')[-1]

        captcha_length = len(captcha_text)
        letters = list(captcha_text)
        letter_images = character_segmenter.get_letter_images(image, captcha_length)

        for i, image in enumerate(letter_images):
            path = f'{target_directory}/{letters[i]}/{time.time()}.png';
            # os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                image.save(path)
            except:
                pass

    end_time = time.time()

    print(f'Completed {len(files)} files in {end_time - start_time} seconds')


if __name__ == '__main__':
    train_from_folder()
