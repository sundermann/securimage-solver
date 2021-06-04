import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

N_Data = 100000

img_rows, img_cols = 32, 32
num_classes = 135
input_shape = (img_rows, img_cols, 1)

def train():
    image_gen = ImageDataGenerator(rescale=1./255)

    train_it = image_gen.flow_from_directory("preprocessing2/train", target_size=(img_rows, img_cols), color_mode="grayscale")
    test_it = image_gen.flow_from_directory("preprocessing2/test", target_size=(img_rows, img_cols), color_mode="grayscale")

    print(train_it.class_indices)

    batch = train_it.next()
    print(batch[0].shape)
    plt.imshow(batch[0][0], cmap="gray")
    plt.show()

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit_generator(generator=train_it,
                        validation_data=test_it,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=2)
    loss = model.evaluate_generator(test_it, steps=24)
    model.save('model.h5')
    print('Test loss:', loss[0])
    print('Test accuracy:', loss[1])

if __name__ == '__main__':
    train()