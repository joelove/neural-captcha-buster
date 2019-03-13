import time
import glob
import cv2
import sys
import numpy as np

from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_model(src_directory='training_images'):
    start_time = time.time()
    print("Creating training data")

    x_train = []
    y_train = []
    image_directories = glob.glob(src_directory + "/*")

    for _, directory in enumerate(image_directories):
        letter = directory.split('/')[-1]
        image_files = glob.glob(directory + "/*.png")

        for _, filename in enumerate(image_files):
            try:
                image = cv2.imread(filename)
                x_train.append(image)
                y_train.append(letter)
            except:
                pass

    print("Training model")

    model = Sequential()

    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dense(26, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, validation_split=0.1, epochs=10, verbose=1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")

    print("Saved model to disk")


if __name__ == '__main__':
    train_model()
