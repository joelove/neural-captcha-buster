from keras.models import model_from_json


loaded_model = None


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    print("Loaded model from disk!")


def read_character(letter_image):
    return loaded_model.predict(letter_image)


def read_characters(letter_images):
    load_model()

    characters = list(map(read_character, letter_images))

    return characters
