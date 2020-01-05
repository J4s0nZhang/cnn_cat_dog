from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from math import ceil
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# create the cnn
classifier = Sequential()

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(128, 128, 3)))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units=64, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# get the data using the image library
from keras.preprocessing.image import ImageDataGenerator
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,  # rescale so pixel values are between 0 and 1
                                   shear_range=0.2,  # random transformations
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory("dataset/training_set", target_size=(128, 128), batch_size=batch_size, class_mode="binary")
testing_set = test_datagen.flow_from_directory("dataset/test_set", target_size=(128, 128), batch_size=batch_size, class_mode="binary")

# fit the classifier:
# you need a navidia gpu what an oof

classifier.fit_generator(training_set,
                         steps_per_epoch= ceil(8000/batch_size),
                         epochs=90,
                         validation_data=testing_set,
                         validation_steps= ceil(2000/batch_size),
                         workers=4)


# save the model architecture into a json file
model_json = classifier.to_json()
json_file = open("Classifier.json", "w")
json_file.write(model_json)
json_file.close()

# save the model weights to a h5py file
model_weights = classifier.save_weights("Classifier.h5")




