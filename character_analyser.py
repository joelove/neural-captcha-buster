import cv2
import numpy as np

from keras.models import model_from_json


def load_model():
    json_file = open('model.json', 'r')
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
        letter_image = letter_image.reshape(48, 48, 1)

        return model.predict(letter_image)

    characters = list(map(read_character, letter_images))

    return characters
