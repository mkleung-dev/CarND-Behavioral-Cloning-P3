import csv
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import math

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D

def readFromFile(folder_name):
    samples = []
    
    with open(os.path.join('data', folder_name, 'driving_log.csv')) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if line[0] == 'center':
                continue
            if line[0].find('\\') >= 0:
                center_file_name = line[0].split('\\')[-1]
                left_file_name = line[0].split('\\')[-1]
                right_file_name = line[0].split('\\')[-1]
            else:
                center_file_name = line[0].split('/')[-1]
                left_file_name = line[0].split('/')[-1]
                right_file_name = line[0].split('/')[-1]

            center_file_name = os.path.join('data', folder_name, 'IMG', center_file_name)
            left_file_name = os.path.join('data', folder_name, 'IMG', left_file_name)
            right_file_name = os.path.join('data', folder_name, 'IMG', right_file_name)
            steering = float(line[3])
            samples.append([center_file_name, left_file_name, right_file_name, steering])
            samples.append([center_file_name, left_file_name, right_file_name, steering])

    return np.array(samples)

def data_generator(samples, batch_size=32):
    multiple_camera_correction = 0.12

    flips = np.zeros((samples.shape[0] * 3, ), dtype=int)
    flips = np.concatenate((flips, np.ones((samples.shape[0] * 3,), dtype=int)), axis=0)

    file_names = np.concatenate((samples[:,0], samples[:,1], samples[:,2]), axis=0)

    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)

    sample_count = steerings.shape[0]
    while True:
        file_names, steerings = sklearn.utils.shuffle(file_names, steerings)
        for offset in range(0, sample_count, batch_size):
            batch_file_names = file_names[offset:offset+batch_size]
            batch_steerings = steerings[offset:offset+batch_size]
            batch_flips = flips[offset:offset+batch_size]
            
            images = []
            for index, file_name in enumerate(batch_file_names):
                image = mpimg.imread(file_name)
                if batch_flips[index] == 1:
                    image = np.fliplr(image)
                    batch_steerings[index] = -batch_steerings[index]
                images.append(image)

            x_train = np.array(images)
            y_train = np.array(batch_steerings)
            yield (x_train, y_train)


def behavioral_cloning_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((40,20), (0,0))))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1))

    return model

#folder_names = ['Normal', 'Reverse', 'sample_driving_data']
#folder_names = ['sample_driving_data']
#folder_names = ['Normal', 'Reverse', 'Normal2', 'Reverse2', 'Turn']
#folder_names = ['track_1_normal_1', 'track_1_reverse_1', 'track_1_normal_2', 'track_1_reverse_2', 'track_1_normal_3', 'track_1_reverse_3', 'turn', 'sharp_turn']
folder_names = ['track_1_normal_4', 'track_1_reverse_4', 'track_1_turn', 'track_1_sharp_turn_1', 'track_1_sharp_turn_2', 'track_1_normal_recovery', 'track_1_reverse_recovery', 'track_1_special_recovery']
for index, folder_name in enumerate(folder_names):
    if index == 0:
        samples = readFromFile(folder_name)
    else:
        temp_samples = readFromFile(folder_name)
        samples = np.concatenate((samples, temp_samples), axis=0)

print(np.min(np.array(samples[:,3], np.float)))
print(np.max(np.array(samples[:,3], np.float)))

train_samples, validation_samples = train_test_split(samples, test_size=0.3)

batch_size = 32

train_data_generator = data_generator(train_samples, batch_size=batch_size)
validation_data_generator = data_generator(validation_samples, batch_size=batch_size)

adam = keras.optimizers.Adam(learning_rate=0.001)
model = behavioral_cloning_model()
model.compile(loss='mse', optimizer=adam)
model.summary()
model.fit_generator(train_data_generator,
steps_per_epoch=math.ceil(len(train_samples)/batch_size),
validation_data=validation_data_generator,
validation_steps=math.ceil(len(validation_samples)/batch_size),
epochs=10, verbose=1)

model.save('model.h5')