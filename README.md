# TensorFlow-Image-Classification
A version of the image classifier built with transfer learning through ResNet50.

### Requirements:
Needs a h5 file with weights named model_weights.h5, a model file named model.json and a json file with the classes named class_names.json. keras documentation on saving model to json and weights to h5: https://keras.io/getting-started/faq/#savingloading-only-a-models-architecture You can generate the class_names json file with this code:
``` python
import json

class_names = json.dumps(<TRAINING_GEN_NAME>.class_indices)
with open("class_names.json", "w") as json_file:
    json_file.write(class_names)
```
For VSCode, add "python.linting.pylintArgs": ["--generate-members"] to settings.json
### Directions:
  Run at least one training sesion to allow the model to create weights and save into the JSON file. This only needs to be done once before predictions should be made.

1. Run GUI.py in the terminal.
2. Either select to upload an image or take a picture.
3. Upload Image:
 a. If the image files didn’t pop up.
 b. Navigate to TensorFlow-Image-Classification/datasets/dogs-vs-cats and select an image from any of the folders.
 c. Click “get prediction”, images do not need to be resized.
4. Take a Picture:
 a. Hold picture up to the camera and press spacebar.
 b. Do not resize images.
 c. Click “get prediction”
