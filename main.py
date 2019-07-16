#Conventional Neural network layering

#Import libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image


model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(units = 128, activation = 'relu'),
    keras.layers.Dense(units = 1, activation = 'sigmoid')
])

#Compiler
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting images
trainDatagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

testDatagen = ImageDataGenerator(rescale = 1./255)

trainingSet = trainDatagen.flow_from_directory(
    'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images',
    target_size = (64, 64),
    class_mode = 'binary'
)

testSet = testDatagen.flow_from_directory(
    'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images',
    target_size = (64, 64),
    class_mode = 'binary'
)

model.fit_generator(
    trainingSet,
    steps_per_epoch = 10,
    epochs = 1,
    validation_data = testSet,
    validation_steps = 80
)
