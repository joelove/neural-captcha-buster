import glob
import cv2
import os
import time
import numpy as np

from PIL import Image

import character_segmenter


def build_training_images(src_directory='solved_captchas', target_directory='training_images'):
    start_time = time.time()
    files = glob.glob(src_directory + "/*.jpg")

    print(f'Creating training images from {len(files)} files...')

    for _, filename in enumerate(files):
        image_array = cv2.imread(filename)

        if not (min(image_array.shape[0:2])):
            print(f'Dropping bad input image: {filename}')
            continue;

        raw_image = Image.fromarray(image_array)
        captcha_text = filename.split('.')[-2].split('_')[-1]

        captcha_length = len(captcha_text)
        letters = list(captcha_text)
        letter_images = character_segmenter.get_letter_images(raw_image, captcha_length)

        if letter_images is None:
            print(f'Failed to segment image: {filename}')
            continue

        for i, image in enumerate(letter_images):
            letter = letters[i]
            directory = f'{target_directory}/{letter}'
            path = f'{directory}/{time.time()}.png';
            try:
                os.makedirs(directory, exist_ok=True)
                image.save(path)
            except:
                print(f'Failed to save image: {path}')
                pass

    end_time = time.time()

    print(f'Completed {len(files)} files in {end_time - start_time} seconds!')


if __name__ == '__main__':
    build_training_images()
