#Conventional Neural network layering

#Import libraries and packages
import keras
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from PIL import Image
import matplotlib.pyplot as plt


#Fitting images
trainDatagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 15,
    zoom_range = 0.2,
    shear_range = 0.2
)

testDatagen = ImageDataGenerator(rescale = 1./255)

trainingSet = trainDatagen.flow_from_directory(
    'datasets/dogs-vs-cats/train',
    target_size = (64, 64),
    class_mode = 'binary'
)

testSet = testDatagen.flow_from_directory(
    'datasets/dogs-vs-cats/test',
    target_size = (64, 64),
    class_mode = 'binary'
)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit_generator(
    trainingSet,
    epochs = 20,
    validation_data = testSet,
    max_queue_size = 25,
    workers = 8,
    shuffle = True
)

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
