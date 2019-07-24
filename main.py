import os, keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, GaussianNoise, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.losses import *
from keras import optimizers
from PIL import Image
import matplotlib.pyplot as plt
from tkinter.filedialog import asksaveasfilename
from tkinter import Tk
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
def combined_loss(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) * 4 + mean_squared_error(y_true, y_pred) * 4 + categorical_hinge(y_true, y_pred) * 2 + categorical_crossentropy(y_true, y_pred)
trainDatagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.0
)
testDatagen = ImageDataGenerator()
checkpoint = ModelCheckpoint("model_weights.h5", monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
trainDatagen.fit(x_train)
testDatagen.fit(x_test)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.2))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(GaussianNoise(0.2))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'sgd', loss = combined_loss, metrics = ['categorical_accuracy'])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
history = model.fit_generator(trainDatagen.flow(x_train, y_train, batch_size = 25), epochs = 10, steps_per_epoch = 2000, validation_data = testDatagen.flow(x_test, y_test, batch_size = 25), max_queue_size = 25, workers = 8, shuffle = True, callbacks = [checkpoint])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print(history.history.keys())
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch ID')
plt.legend(['Training Set', 'Validation Set'], loc='upper left')
plt.show()