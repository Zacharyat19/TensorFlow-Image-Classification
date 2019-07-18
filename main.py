#Using keras to create a conventional neural network through layering

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


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1,save_best_only=True,mode="max")

#Fitting images
#Resizing images to 64 x 64
trainDatagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    rotation_range = 15
)

testDatagen = ImageDataGenerator(rescale = 1./255)

#link data directory
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

#Initialize Sequential model
model = Sequential()
#Start layering with Conv2D, starting at 32 layers
#and inceasing by a factor of two
model.add(Conv2D(64, (3,3), input_shape = (64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = optimizers.SGD(lr = 0.0001, decay = 0, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.load_weights(checkpoint_path)

model.fit_generator(
    trainingSet,
    steps_per_epoch = 781,
    epochs = 20,
    validation_data = testSet,
    validation_steps = 10,
    callbacks = [cp_callback],
    workers = 8,
    max_queue_size = 25,
    shuffle = True
)

