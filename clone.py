import csv
import cv2

lines=[]

dirname = '../behavorial-cloning-data'
with open(dirname + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)

images = []
measurements = []

count = 0
for line in lines:
    source_path = line[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    local_path = "dirname/IMG/" + filename
    image = cv2.imread(local_path)
    images.append(image)
    measurement = line[3]
    measurements.append(measurement)

import numpy as np

x_train = np.array(images)
y_train = np.array(measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True)

model.save(model.h5)
