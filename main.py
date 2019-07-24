#Image classification using ResNet50 transfer learning

#Import required libraries and packages
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import optimizers
from keras.applications import ResNet50
from keras.models import Sequential
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
train_gen = data_gen.flow_from_directory(
    'datasets/dogs-vs-cats/train',
    target_size = (224, 224),
    class_mode = 'categorical'
)

#pull data from dirextory and resize images
valid_gen = data_gen.flow_from_directory(
    'datasets/dogs-vs-cats/Validation',
    target_size = (224, 224),
    class_mode = 'categorical'
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

#Initialize epochs, data, and steps
history = model.fit_generator(
    train_gen,
    epochs = 1,
    steps_per_epoch = 200,
    validation_data = valid_gen,
    validation_steps = 8,
    max_queue_size = 25,
    workers = 4,
    shuffle = True,
)

#make a prediciton based on an image file
img = image.load_img('datasets/dogs-vs-cats/test/Dogs/90.jpg', target_size = (224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
images = np.vstack([x])
classes = model.predict(images)
if classes[0][0] >= 0.98:
    prediction = "dog"
else:
    prediction = "cat"
print(prediction)

#print history
print(history.history.keys())

#Upload history into accuracy graph
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#Upload history into loss graph
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.show()