import time
import glob
import cv2
import sys
import numpy as np

from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def create_model(x_train, y_train):
    print("Training model")
    model = Sequential()
    model.add(Conv2D(48, (5, 5), padding="same", input_shape=(48, 48, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(26, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(np.array(x_train), np.array(y_train), validation_split=0.1, epochs=10, verbose=1)
    save_model(model)


def train_model(src_directory='training_images'):
    print("Creating training data")
    start_time = time.time()
    x_train = []
    y_train = []
    image_directories = glob.glob(src_directory + "/*")

    for _, directory in enumerate(image_directories):
        letter = directory.split('/')[-1]
        image_files = glob.glob(directory + "/*.png")
        letter_images = []

        for _, filename in enumerate(image_files):
            try:
                image = cv2.imread(filename)
                letter_images.append(image)
            except:
                pass

        x_train.append(np.array(letter_images))
        y_train.append(letter)

    create_model(x_train, y_train)


if __name__ == '__main__':
    train_model()
