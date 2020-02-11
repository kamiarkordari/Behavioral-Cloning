import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


lines=[]

#dirname = '../behavorial-cloning-data'
dirname = '/opt/carnd_p3/data'

with open(dirname + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header

    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = dirname+"/IMG/" + filename
        image = cv2.imread(local_path)
        #image_half_size = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        #images.append(image_half_size)
        images.append(image)

    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = -1.0 * float(measurement)
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)


x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#x_train, y_train = shuffle(x_train, y_train)


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

print(x_train.shape)

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(80,160,3)))
# Crop images
model.add(Cropping2D(cropping=((50,20), (0,0))))
#model.add(Cropping2D(cropping=((25,10), (0,0))))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
#model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
