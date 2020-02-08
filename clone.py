import csv
import cv2

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
        images.append(image)

    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

import numpy as np

x_train = np.array(images)
y_train = np.array(measurements)


import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
