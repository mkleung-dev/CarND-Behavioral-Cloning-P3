import csv
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import math
import cv2

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Cropping2D

# Read data
def read_from_file(folder_name):
    samples = []
    
    steering_offset = 0

    if folder_name.find('left_recovery') >= 0:
        steering_offset = float(folder_name.split('_')[-1]) / 100.0

    elif folder_name.find('right_recovery') >= 0:
        steering_offset = -float(folder_name.split('_')[-1]) / 100.0

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
            steering = float(line[3]) + steering_offset
            samples.append([center_file_name, left_file_name, right_file_name, steering])
            samples.append([center_file_name, left_file_name, right_file_name, steering])

    return np.array(samples)

# Compute sample count for Track 1
def get_sample_count_1(samples):
    multiple_camera_correction = 0.15

    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)
    steerings = np.concatenate((steerings, -steerings), axis=0)

    #filtering
    filtered_steerings = steerings[np.abs(steerings) >= 0.2]
    filtered_0_steerings = steerings[np.abs(steerings) <= 0.2]
    filtered_0_steerings = filtered_0_steerings[0:filtered_steerings.shape[0]]

    filtered_steerings = np.concatenate((filtered_steerings, filtered_0_steerings), axis=0)

    sample_count = filtered_steerings.shape[0]
    return sample_count

# Compute sample count for Track 1 and Track 2
def get_sample_count_2(samples):
    multiple_camera_correction = 0.15

    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)
    steerings = np.concatenate((steerings, -steerings), axis=0)

    # Balancing data
    count_02 = (0.2 > np.abs(steerings)).sum()
    count_04 = ((0.4 > np.abs(steerings)) & (np.abs(steerings) >= 0.2)).sum()
    count_06 = ((0.6 > np.abs(steerings)) & (np.abs(steerings) >= 0.4)).sum()
    count_08 = ((0.8 > np.abs(steerings)) & (np.abs(steerings) >= 0.6)).sum()
    count_10 = (np.abs(steerings) >= 0.8).sum()

    min_count = min(count_02, count_04, count_06, count_08, count_10)

    filtered_steerings = steerings[0.2 > np.abs(steerings)][:min_count]
    filtered_steerings = np.concatenate((filtered_steerings, steerings[(0.4 > np.abs(steerings)) & (np.abs(steerings) >= 0.2)][:min_count]), axis=0)
    filtered_steerings = np.concatenate((filtered_steerings, steerings[(0.6 > np.abs(steerings)) & (np.abs(steerings) >= 0.4)][:min_count]), axis=0)
    filtered_steerings = np.concatenate((filtered_steerings, steerings[(0.8 > np.abs(steerings)) & (np.abs(steerings) >= 0.6)][:min_count]), axis=0)
    filtered_steerings = np.concatenate((filtered_steerings, steerings[np.abs(steerings) >= 0.8][:min_count]), axis=0)

    sample_count = filtered_steerings.shape[0]
    return sample_count
    
# Data generator for Track 1
def data_generator_1(samples, batch_size=32):
    multiple_camera_correction = 0.15

    flips = np.zeros((samples.shape[0] * 3, ), dtype=int)
    flips = np.concatenate((flips, np.ones((samples.shape[0] * 3,), dtype=int)), axis=0)

    file_names = np.concatenate((samples[:,0], samples[:,1], samples[:,2]), axis=0)
    file_names = np.concatenate((file_names, samples[:,0], samples[:,1], samples[:,2]), axis=0)

    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)
    steerings = np.concatenate((steerings, -steerings), axis=0)
    
    while True:
        # Shuffling
        file_names, steerings, flips = sklearn.utils.shuffle(file_names, steerings, flips)

        # Balancing data
        filtered_file_names = file_names[np.abs(steerings) >= 0.2]
        filtered_steerings = steerings[np.abs(steerings) >= 0.2]
        filtered_flips = flips[np.abs(steerings) >= 0.2]

        filtered_0_file_names = file_names[np.abs(steerings) <= 0.2]
        filtered_0_steerings = steerings[np.abs(steerings) <= 0.2]
        filtered_0_flips = flips[np.abs(steerings) <= 0.2]

        filtered_0_file_names = filtered_0_file_names[0:filtered_file_names.shape[0]]
        filtered_0_steerings = filtered_0_steerings[0:filtered_steerings.shape[0]]
        filtered_0_flips = filtered_0_flips[0:filtered_flips.shape[0]]

        filtered_file_names = np.concatenate((filtered_file_names, filtered_0_file_names), axis=0)
        filtered_steerings = np.concatenate((filtered_steerings, filtered_0_steerings), axis=0)
        filtered_flips = np.concatenate((filtered_flips, filtered_0_flips), axis=0)

        # Shuffling
        filtered_file_names, filtered_steerings, filtered_flips = sklearn.utils.shuffle(filtered_file_names, filtered_steerings, filtered_flips)

        sample_count = filtered_steerings.shape[0] 

        for offset in range(0, sample_count, batch_size):
            batch_file_names = filtered_file_names[offset:offset+batch_size]
            batch_steerings = filtered_steerings[offset:offset+batch_size]
            batch_flips = filtered_flips[offset:offset+batch_size]
            
            images = []
            for index, file_name in enumerate(batch_file_names):
                image = mpimg.imread(file_name)
                if batch_flips[index] == 1:
                    image = np.fliplr(image)
                images.append(image)

            x_train = np.array(images)
            y_train = np.array(batch_steerings)
            yield (x_train, y_train)

# Data generator for Track 1 and Track 2
def data_generator_2(samples, batch_size=32):
    multiple_camera_correction = 0.15

    flips = np.zeros((samples.shape[0] * 3, ), dtype=int)
    flips = np.concatenate((flips, np.ones((samples.shape[0] * 3,), dtype=int)), axis=0)

    file_names = np.concatenate((samples[:,0], samples[:,1], samples[:,2]), axis=0)
    file_names = np.concatenate((file_names, samples[:,0], samples[:,1], samples[:,2]), axis=0)

    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)
    steerings = np.concatenate((steerings, -steerings), axis=0)
    
    while True:
        # Shuffling
        file_names, steerings, flips = sklearn.utils.shuffle(file_names, steerings, flips)

        # Balancing data
        count_02 = (0.2 > np.abs(steerings)).sum()
        count_04 = ((0.4 > np.abs(steerings)) & (np.abs(steerings) >= 0.2)).sum()
        count_06 = ((0.6 > np.abs(steerings)) & (np.abs(steerings) >= 0.4)).sum()
        count_08 = ((0.8 > np.abs(steerings)) & (np.abs(steerings) >= 0.6)).sum()
        count_10 = (np.abs(steerings) >= 0.8).sum()

        min_count = min(count_02, count_04, count_06, count_08, count_10)

        filtered_file_names = file_names[0.2 > np.abs(steerings)][:min_count]
        filtered_steerings = steerings[0.2 > np.abs(steerings)][:min_count]
        filtered_flips = flips[0.2 > np.abs(steerings)][:min_count]
        
        filtered_file_names = np.concatenate((filtered_file_names, file_names[(0.4 > np.abs(steerings)) & (np.abs(steerings) >= 0.2)][:min_count]), axis=0)
        filtered_steerings = np.concatenate((filtered_steerings, steerings[(0.4 > np.abs(steerings)) & (np.abs(steerings) >= 0.2)][:min_count]), axis=0)
        filtered_flips = np.concatenate((filtered_flips, flips[(0.4 > np.abs(steerings)) & (np.abs(steerings) >= 0.2)][:min_count]), axis=0)

        filtered_file_names = np.concatenate((filtered_file_names, file_names[(0.6 > np.abs(steerings)) & (np.abs(steerings) >= 0.4)][:min_count]), axis=0)
        filtered_steerings = np.concatenate((filtered_steerings, steerings[(0.6 > np.abs(steerings)) & (np.abs(steerings) >= 0.4)][:min_count]), axis=0)
        filtered_flips = np.concatenate((filtered_flips, flips[(0.6 > np.abs(steerings)) & (np.abs(steerings) >= 0.4)][:min_count]), axis=0)
        
        filtered_file_names = np.concatenate((filtered_file_names, file_names[(0.8 > np.abs(steerings)) & (np.abs(steerings) >= 0.6)][:min_count]), axis=0)
        filtered_steerings = np.concatenate((filtered_steerings, steerings[(0.8 > np.abs(steerings)) & (np.abs(steerings) >= 0.6)][:min_count]), axis=0)
        filtered_flips = np.concatenate((filtered_flips, flips[(0.8 > np.abs(steerings)) & (np.abs(steerings) >= 0.6)][:min_count]), axis=0)
        
        filtered_file_names = np.concatenate((filtered_file_names, file_names[np.abs(steerings) >= 0.8][:min_count]), axis=0)
        filtered_steerings = np.concatenate((filtered_steerings, steerings[np.abs(steerings) >= 0.8][:min_count]), axis=0)
        filtered_flips = np.concatenate((filtered_flips, flips[np.abs(steerings) >= 0.8][:min_count]), axis=0)

        # Shuffling
        filtered_file_names, filtered_steerings, filtered_flips = sklearn.utils.shuffle(filtered_file_names, filtered_steerings, filtered_flips)

        sample_count = filtered_steerings.shape[0] 

        for offset in range(0, sample_count, batch_size):
            batch_file_names = filtered_file_names[offset:offset+batch_size]
            batch_steerings = filtered_steerings[offset:offset+batch_size]
            batch_flips = filtered_flips[offset:offset+batch_size]
            
            images = []
            for index, file_name in enumerate(batch_file_names):
                image = mpimg.imread(file_name)
                if batch_flips[index] == 1:
                    image = np.fliplr(image)
                images.append(image)

            x_train = np.array(images)
            y_train = np.array(batch_steerings)
            yield (x_train, y_train)

# Model by Nvidia
def pilot_net_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((40,20), (0,0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model

# My proposed model
def behavioral_cloning_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
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
    model.add(Dropout(0.8))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))

    return model

# Train model for Track 1
def run_model_for_track_1(learning_rate, epoch, output_file_name):
    folder_names = ['track_1_normal', 'track_1_reverse',
    'track_1_normal_left_recovery_015', 'track_1_normal_right_recovery_015',
    'track_1_reverse_left_recovery_015', 'track_1_reverse_right_recovery_015']

    for index, folder_name in enumerate(folder_names):
        if index == 0:
            samples = read_from_file(folder_name)
        else:
            temp_samples = read_from_file(folder_name)
            samples = np.concatenate((samples, temp_samples), axis=0)

    samples = sklearn.utils.shuffle(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.3)

    batch_size = 32

    train_data_generator = data_generator_1(train_samples, batch_size=batch_size)
    validation_data_generator = data_generator_1(validation_samples, batch_size=batch_size)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model = behavioral_cloning_model()
    model.compile(loss='mse', optimizer=adam)
    model.summary()

    history = model.fit_generator(train_data_generator,
    steps_per_epoch=math.ceil(get_sample_count_1(train_samples)/batch_size),
    validation_data=validation_data_generator,
    validation_steps=math.ceil(get_sample_count_1(validation_samples)/batch_size),
    epochs=epoch, verbose=1)

    model.save('{}.h5'.format(output_file_name))

    with open('{}.npy'.format(output_file_name), 'wb') as f:
        np.save(f, np.array(history.history['loss']))
        np.save(f, np.array(history.history['val_loss']))

# Train model for Track 1 and Track 2
def train_model_for_track_1_and_track_2(learning_rate, epoch, output_file_name):
    folder_names = [
        'track_1_normal', 'track_1_reverse',
        'track_1_normal_left_recovery_015', 'track_1_normal_right_recovery_015',
        'track_1_reverse_left_recovery_015', 'track_1_reverse_right_recovery_015',
        'track_2_normal_1','track_2_normal_2',
        'track_2_reverse_1','track_2_reverse_2',
        'track_2_normal_fantasy_1', 'track_2_normal_fantasy_2',
        'track_2_reverse_fantasy_1', 'track_2_reverse_fantasy_2',
        'track_2_left_normal_1','track_2_left_normal_2',
        'track_2_left_reverse_1','track_2_left_reverse_2',
        'track_2_left_normal_fantasy_1', 'track_2_left_normal_fantasy_2',
        'track_2_left_reverse_fantasy_1', 'track_2_left_reverse_fantasy_2',
        'track_2_special_1', 'track_2_special_1', 
        'track_2_special_fantasy_1', 'track_2_special_fantasy_2', 'track_2_special_fantasy_3']

    for index, folder_name in enumerate(folder_names):
        if index == 0:
            samples = read_from_file(folder_name)
        else:
            temp_samples = read_from_file(folder_name)
            samples = np.concatenate((samples, temp_samples), axis=0)

    samples = sklearn.utils.shuffle(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.3)

    batch_size = 32

    train_data_generator = data_generator_2(train_samples, batch_size=batch_size)
    validation_data_generator = data_generator_2(validation_samples, batch_size=batch_size)

    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    model = behavioral_cloning_model()
    model.compile(loss='mse', optimizer=adam)
    model.summary()

    history = model.fit_generator(train_data_generator,
    steps_per_epoch=math.ceil(get_sample_count_2(train_samples)/batch_size),
    validation_data=validation_data_generator,
    validation_steps=math.ceil(get_sample_count_2(validation_samples)/batch_size),
    epochs=epoch, verbose=1)

    model.save('{}.h5'.format(output_file_name))

    with open('{}.npy'.format(output_file_name), 'wb') as f:
        np.save(f, np.array(history.history['loss']))
        np.save(f, np.array(history.history['val_loss']))


# Train model for Track 1
run_model_for_track_1(0.0001, 10, 'model_track_1_temp')

# Study learning rate
train_model_for_track_1_and_track_2(0.001, 30, 'model_1_and_2_0001')
train_model_for_track_1_and_track_2(0.0001, 30, 'model_1_and_2_00001')
train_model_for_track_1_and_track_2(0.00001, 30, 'model_1_and_2_000001')

# Train model for Track 1 and Track 2
train_model_for_track_1_and_track_2(0.0001, 20, 'model')
