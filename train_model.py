import glob
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def save_model(model):
    with open("model.json", "w") as json_file:
        json_file.write(model.to_json())

    model.save_weights("model.h5")


def create_model(X_train, Y_train):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(26, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, Y_train, validation_split=0.1, epochs=1, verbose=1)

    return model


def train_model(src_directory="training_images"):
    image_directories = glob.glob(src_directory + "/*")

    X_train = []
    Y_train = []

    for _, directory in enumerate(image_directories):
        image_files = glob.glob(directory + "/*.png")

        letter = directory.split("/")[-1]
        letter_vector = np.zeros(26)
        letter_vector[ord(letter) - ord("a")] = 1

        for _, filename in enumerate(image_files):
            letter_image = cv2.imread(filename)
            letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
            letter_image = letter_image.reshape(letter_image.shape + (1,))
            X_train.append(letter_image)
            Y_train.append(letter_vector)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    model = create_model(X_train, Y_train)

    save_model(model)


if __name__ == "__main__":
    train_model()
