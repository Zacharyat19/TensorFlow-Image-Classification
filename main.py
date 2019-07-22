import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

NUM_CLASSES = 2
CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
BATCH_SIZE_TESTING = 1
VAL_SPLIT = 0.3

Image_width = 500
Image_height = 374

train_dir = 'datasets/dogs-vs-cats/train'

model = Sequential()
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = 'imagenet'))
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
model.layers[0].trainable = False
model.summary()

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

image_size = IMAGE_RESIZE

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_image_gen = ImageDataGenerator(rescale=1/255,validation_split = VAL_SPLIT)

train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width,Image_height),
    batch_size=BATCH_SIZE_TRAINING,
    seed=42,
    subset='training',
    shuffle=True
)

validation_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width,Image_height),
    batch_size=BATCH_SIZE_VALIDATION,
    seed=42,
    subset='validation',
    shuffle=True
)


model = Sequential()
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE,
weights = 'imagenet'))
model.layers[0].trainable = False
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = optimizers.SGD(lr = 0.0001, decay = 0, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(
    trainingSet,
    epochs = 20,
    steps_per_epoch = 781,
    validation_data = testSet,
    validation_steps = 10,
    max_queue_size = 25,
    workers = 8,
    shuffle = True,
    callbacks = [cp_callback]
)
model.load_weights("../working/best.hdf5")

print(fit_history.history.keys())

plt.figure(1, figsize = (15,8))

plt.subplot(221)
plt.plot(fit_history.history['acc'])
plt.plot(fit_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.show()
