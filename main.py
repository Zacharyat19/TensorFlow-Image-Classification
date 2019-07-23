import os, keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation, GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras import optimizers
from PIL import Image
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from keras.datasets import cifar100
from keras.utils.np_utils import to_categorical

Tk().withdraw()
filename = askopenfilename()
checkpoint_dir = os.path.dirname(filename)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filename, save_weights_only = True, verbose = 1, save_best_only = True)
trainDatagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
#        rotation_range=15,
#        width_shift_range=0.1,
#        height_shift_range=0.1,
#        shear_range=0.2,
#        zoom_range=0.2,
#        channel_shift_range=0.1,
#        fill_mode='nearest',
#        horizontal_flip=True,
#        vertical_flip=True,
        validation_split=0.0
)
testDatagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False
)
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
trainDatagen.fit(x_train)
testDatagen.fit(x_test)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='softplus', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.04))
model.add(Conv2D(64, (3, 3), activation='softplus', kernel_initializer='he_uniform', padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.04))
model.add(Conv2D(128, (3, 3), activation='softplus', kernel_initializer='he_uniform', padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.04))
model.add(Conv2D(256, (3, 3), activation='softplus', kernel_initializer='he_uniform', padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.04))
model.add(Conv2D(512, (3, 3), activation='softplus', kernel_initializer='he_uniform', padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(GaussianNoise(0.04))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('softplus'))
model.add(GaussianNoise(0.04))
model.add(Dense(100, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'categorical_hinge', metrics = ['categorical_accuracy'])
history = model.fit_generator(trainDatagen.flow(x_train, y_train, batch_size = 25), epochs = 500, steps_per_epoch = 2000, validation_data = testDatagen.flow(x_test, y_test, batch_size = 25), validation_steps = 10,
    max_queue_size = 25, workers = 8, shuffle = True, callbacks = [cp_callback]
)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()