# image classification problem practice
# basic training of cats and dogs classification
# test set is 2000, training set is 8000 images, categories divided by folder name
# 80% to 20% split between test and training set

# with this setup we don't need to encode any categorical data like with ann
# and we already split into training and test set
# but we still need to do feature scaling
# with images the data pre-processing step is done manually

# part1 - building the cnn

# import sequential style of network creation and the layers we need
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# initialize cnn in keras (called dogcatClassifier as part of the sequential class)
dogcatClassifier = Sequential()

# let's add the convolution layer to the class
# convolution 2D takes a couple of parameters (# of feature maps, number of rows and columns of the filter)
# conventional to start with 32 feature detectors 3x3 dimensions, border mode = same is fine, input shape
# (this means we'll need to force our image into the same size)
# for input_shape, 3 channels because we have a coloured image, and we want to keep it, and the other two are dimensions
# of the image
# stride is 1
dogcatClassifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu')) # this order of input_shape input is for tensorflow backend

# let's add the maxpooling layer
# maxpooling is a stride of 2 (size of feature map is divided by 2)
# in general maxpooling window size is 2x2
dogcatClassifier.add(MaxPool2D(pool_size=(2, 2)))

# let's add the flattening layer so we have our input
# we don't lose spacial structure from flattening, because the high numbers in the feature maps
# represent the spacial structure and we keep these
dogcatClassifier.add(Flatten())

# let's make the classic ann for actual classification
# the flatten layer is our input layer
# output_dim is the number of nodes for the hidden layer, common practice is to choose hidden nodes between input and output
# the number of nodes should be found with experiementation (good practice to pick a power of 2
# first we'll add the hidden layer
dogcatClassifier.add(Dense(activation= 'relu', units=128))

# now we'll add the output layer
# since we have a binary classification, we only need one node and we can use the sigmoid activation function
dogcatClassifier.add(Dense(activation= 'sigmoid', units=1))

# now we need to add gradient descent and loss functions
# we use binary cross entropy because we have binary classification and we use logarithmic loss in general for
# classification problems and cross entropy corresponds to logarithmic loss
# performance metric is the accuracy metric
dogcatClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])


# now lets fit the cnn to our images
# part 2 - fitting the cnn to images
# image pre processing is good for preventing overfitting
# take the keras documentation shortcut for the code for image pre-processing

from keras.preprocessing.image import ImageDataGenerator
from math import ceil
train_datagen = ImageDataGenerator( # function lets us do image augmentation
        rescale=1./255,  # rescale so pixel values are between 0 and 1
        shear_range=0.2,  # random transformations
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # only rescale pixels for the test set

training_set = train_datagen.flow_from_directory( # creates the training set
        'dataset/training_set',
        target_size=(64, 64), # the size of the images
        batch_size=32,         # size of the batches (32 images will go through the cnn before weights are updated)
        class_mode='binary')   # only two classes so we can keep binary

test_set = test_datagen.flow_from_directory( # creates the test set
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

dogcatClassifier.fit_generator( # fit generator is what we use to train our model
        training_set,
        steps_per_epoch= ceil(8000/32), # size of our training set
        epochs=25,             # number of times we run through the entire training set once
        validation_data=test_set,
        validation_steps= ceil(2000/32)) # number of images in our test set

# ok now that we've trained our network, lets use it predict some new images
# now if we think about it, the weights of the network during training are gone now if we don't save the network
# oh well, save it later lets just retrain everytime like a goon

import numpy as np
from keras.preprocessing import image

# first let's read and properly convert the two single prediction images
# load the images and set their dims to 64 by 64
image1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
image2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))

input1 = image.img_to_array(image2)  # this will generate our 3 dimensional array (separated
# add one more dimension for the predict method to work, the 4th dimension is for the batch
input1 = np.expand_dims(input1, axis=0)
result = dogcatClassifier.predict(input1)  # get the numerical result from our model
if result[0][0] == 1:# training_set.class_indices will tell us how the numerical result corresponds to our actual labels
    print("dog")
else:
    print("cat")


# let's now save our model to json to use for later:
model_json = dogcatClassifier.to_json()
with open('dogcatClassifier.json', 'w') as json_file:
    json_file.write(model_json)


# let's save the trained weights so we can load it into the network
dogcatClassifier.save_weights("dogcatClassifier.h5")






