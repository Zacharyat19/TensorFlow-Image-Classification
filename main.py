#Image classification using ResNet50 transfer learning

#Import required libraries and packages
import keras
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import optimizers
from keras.applications import ResNet50
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

#classification can either be Cat or Dog
NUM_CLASSES = 2

#Augmented training data allows a greater sample of data to use
train_data_gen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

data_gen = ImageDataGenerator(preprocessing_function = preprocess_input)

#pull data from dirextory and resize images
train_gen = train_data_gen.flow_from_directory(
    'datasets/dogs-vs-cats/train',
    target_size = (224, 224),
    class_mode = 'categorical'
)

class_names = json.dumps(train_gen.class_indices)
with open("class_names.json", "w") as json_file:
    json_file.write(class_names)

#pull data from dirextory and resize images
valid_gen = data_gen.flow_from_directory(
    'datasets/dogs-vs-cats/Validation',
    target_size = (224, 224),
    class_mode = 'categorical',
    shuffle = False
)

#Initialize model
model = Sequential()

#Add ResNet50 as first layer
model.add(ResNet50(include_top = False, pooling = 'avg',
weights = 'Weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))

#Create a final Dense layer for our model
model.add(Dense(NUM_CLASSES, activation = 'softmax'))

#Tell the model not to train first layer since it's already trained
model.layers[0].trainable = False

#Compiler
sgd = optimizers.SGD(lr = 0.01, decay = 0, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

#Generate epochs, steps, and data
#Uses history the function of Keras for graphing data
history = model.fit_generator(
    train_gen,
    epochs = 20,
    steps_per_epoch = 25,
    validation_data = valid_gen,
    validation_steps = 8,
    max_queue_size = 25,
    workers = 4,
    shuffle = True,
)

#Save weights
model.save_weights("Weights/model_weights.h5")

#print history
print(history.history.keys())

#Upload history into accuracy graph
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')

#Upload history into loss graph
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.show()

#Convert model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)