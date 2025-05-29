# Signify
An image classifier built using TensorFlow and Keras API.

If you wish to compile the application yourself, install pyinstaller and run ```pyinstaller -w -F -n "Signify" -i cats-dogs.ico main.py``` from the main directory. Once it finishes building the executable will be in the dist folder. Otherwise the python instructions are down below.

### Requirements:
Needs a h5 file with weights named model_weights.h5, a model file named model.json and a json file with the classes named class_names.json. Some premade models are located in the models folder. Drag the three required files into the same directory as either main.py (if you run the python file) or the "Signify" executable (if you compile it yourself). keras documentation on saving model to json and weights to h5 can be found [here](https://keras.io/getting-started/faq/#savingloading-only-a-models-architecture). You can generate the class_names json file with this code:
```python
import json

class_names = json.dumps(TRAINING_GEN_NAME.class_indices)
with open("class_names.json", "w") as json_file:
    json_file.write(class_names)
```
If the code to generate class_names.json is unable to run, the format for class_names is a dict formatted like this: {"CLASS_0_NAME":0,"CLASS_1_NAME":1,"CLASS_2_NAME":2}

### Directions:
1. Make sure you have met the requirements mentioned above.
2. Either select an image(.jpg, .png, .gif, .ppm and .ico files), or take a picture (press space in the take picture dialogue box).
3. Enter the picture size that your model takes in (default is 64x64).
4. Press the 'get prediction' button to get your prediction.

### Python instructions:
1. Make sure you are using python 3.6.8. The download is available [here](https://www.python.org/downloads/release/python-368/).
2. Run ```pip install requirements.txt``` (if you wish to use your gpu for training with tensorflow, look at https://www.tensorflow.org/install/gpu)
3. Make sure the requirements are fufilled.
4. Run main.py either through idle or from cmd by navigating to it's directory and running ```python main.py```.
