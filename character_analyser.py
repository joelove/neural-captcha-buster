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
        return model.predict(letter_image)

    characters = list(map(read_character, letter_images))

    return characters
