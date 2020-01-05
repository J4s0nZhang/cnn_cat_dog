from keras.models import model_from_json
from keras.models import Sequential
import numpy as np
from keras.preprocessing import image

# load model by reading the json file
json_file = open("dogcatClassifier.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# create the model from the json architecture file
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("dogcatClassifier.h5")


test_datagen = image.ImageDataGenerator(rescale=1./255) # only rescale pixels for the test set
test_set = test_datagen.flow_from_directory("dataset/single_prediction", target_size=(64, 64), batch_size=6, class_mode="binary", shuffle=False)
result = loaded_model.predict_generator(test_set, steps=1)

index = 0
while index < len(result):
    print("prediction: " + str(result[index][0]))
    if result[index][0] > 0.5:
        print("dog")
    else:
        print("cat")


    index = index + 1

