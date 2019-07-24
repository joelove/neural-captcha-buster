from cv2 import cv2
from keras.models import model_from_json

import numpy as np


def load_model():
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    return loaded_model


def read_characters(letter_images):
    model = load_model()

    def read_character(letter_image):
        letter_image = np.array(letter_image)
        letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
        letter_image = letter_image.reshape((1,) + letter_image.shape + (1,))
        letter_image = letter_image / 255.0

        prediction = model.predict(letter_image)

        return prediction[0]

    characters = list(map(read_character, letter_images))

    return characters
