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
NUM_EPOCHS = 1
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = 100
BATCH_SIZE_VALIDATION = 100
BATCH_SIZE_TESTING = 1
VAL_SPLIT = 0.3

model = Sequential()
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE,
weights = 'imagenet'))
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
model.layers[0].trainable = False
model.summary()

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

image_size = IMAGE_RESIZE

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_image_gen = ImageDataGenerator(rescale=1/255,validation_split = VAL_SPLIT)

train_generator = train_image_gen.flow_from_directory(
    'datasets/dogs-vs-cats/train',
    target_size = (224, 224),
    batch_size = BATCH_SIZE_TRAINING,
    seed=42,
    subset ='training',
    shuffle = True
)

val_generator = data_generator.flow_from_directory(
    'datasets/dogs-vs-cats/test',
    target_size = (224, 224),
    batch_size = BATCH_SIZE_TESTING,
    seed=42,
    subset ='validation',
    shuffle = True
)

(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(val_generator))

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5', monitor = 'val_loss', save_best_only = True, mode = 'auto')

fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch = STEPS_PER_EPOCH_TRAINING,
        epochs = NUM_EPOCHS,
        validation_data = val_generator,
        validation_steps = STEPS_PER_EPOCH_VALIDATION,
        callbacks = [cb_checkpointer, cb_early_stopper]
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
