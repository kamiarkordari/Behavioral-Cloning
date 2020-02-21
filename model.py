import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.models import Model
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import os
import math


dirname = '/opt/carnd_p3/data'

lines=[]
samples = []
with open(dirname + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    for line in reader:
        lines.append(line) #
        samples.append(line)

sklearn.utils.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# use generator for memory efficiency by processing data in batches
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # loop to read center, left, and right images
                for i in range(3):
                    source_path = batch_sample[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = dirname+"/IMG/" + filename
                    image = cv2.imread(local_path)
                    images.append(image)

                # add steering data for each image by correcting values for right and left images
                correction = 0.2
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                measurements.append(measurement+correction)
                measurements.append(measurement-correction)

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                # flip each image and add it to the dataset
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = -1.0 * float(measurement)
                augmented_images.append(flipped_image)
                augmented_measurements.append(flipped_measurement)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# set batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# size of the input image to model
ch, row, col = 3, 160, 320

# create model - NVDIA's end-to-end autonomous car architecture
model = Sequential()
# preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# crop the top and bottom section of each image
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=3, verbose=1)


model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
