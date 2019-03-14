import glob
import cv2
import os
import time

from PIL import Image

import character_segmenter


def build_training_images(src_directory='solved_captchas', target_directory='training_images'):
    start_time = time.time()
    files = glob.glob(src_directory + "/*.jpg")

    print(f'Creating training images from {len(files)} files')

    for _, filename in enumerate(files):
        image = Image.fromarray(cv2.imread(filename))
        captcha_text = filename.split('.')[-2].split('_')[-1]

        captcha_length = len(captcha_text)
        letters = list(captcha_text)
        letter_images = character_segmenter.get_letter_images(image, captcha_length)

        if letter_images is None:
            print(f'Dropping image: {captcha_text}')
            break

        for i, image in enumerate(letter_images):
            letter = letters[i]
            directory = f'{target_directory}/{letter}'
            path = f'{directory}/{time.time()}.png';
            try:
                os.makedirs(directory, exist_ok=True)
                image.save(path)
            except:
                print(f'Failed to save: {path}')
                pass

    end_time = time.time()

    print(f'Completed {len(files)} files in {end_time - start_time} seconds')


if __name__ == '__main__':
    build_training_images()
