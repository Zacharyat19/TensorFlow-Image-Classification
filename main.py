#Conventional Neural network layering

#Import libraries and packages
import keras
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

#checkpoint_path = "training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

#model.load_weights(checkpoint_path)

# Create checkpoint callback
#cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

#Compiler


#Fitting images
trainDatagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
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
model.add(Conv2D(64, (3,3), input_shape = (64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit_generator(
    trainingSet,
    steps_per_epoch = 50,
    epochs = 50,
    validation_data = testSet,
    validation_steps = 150,
    #callbacks = [cp_callback]
)
