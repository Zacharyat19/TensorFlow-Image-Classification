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

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1,save_best_only=True,mode="max")

trainDatagen = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

testDatagen = ImageDataGenerator(
    rescale=1./255
)

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

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
