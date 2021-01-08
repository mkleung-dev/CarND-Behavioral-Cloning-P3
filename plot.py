import csv
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import math
import cv2

# Read data sample
def readFromFile(folder_name):
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

# Load training history
def load_history(file_name):
    with open('{}.npy'.format(file_name), 'rb') as f:
        loss = np.load(f)
        val_loss = np.load(f)

        return loss, val_loss

# Plot the model loss against learning rate
def plot_model_loss_against_learning_rate():
    loss_0001, val_loss_0001 = load_history('model_1_and_2_0001')
    loss_00001, val_loss_00001 = load_history('model_1_and_2_00001')
    loss_000001, val_loss_000001 = load_history('model_1_and_2_000001')

    # Plot data
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 5))
    fig.suptitle('Model Loss')

    axs[0].plot(loss_0001)
    axs[0].plot(val_loss_0001)
    axs[0].legend(["Training Loss", "Validation Loss"],loc="upper right")
    axs[0].set_title('Adam with learning rate = 0.001')

    print('-' * 60)
    print('Adam with learning rate = 0.001')
    print('Min Loss: {}'.format(np.min(loss_0001)))
    print('Min Validation Loss: {}'.format(np.min(val_loss_0001)))

    axs[1].plot(loss_00001)
    axs[1].plot(val_loss_00001)
    axs[1].legend(["Training Loss", "Validation Loss"],loc="upper right")
    axs[1].set_title('Adam with learning rate = 0.0001')

    print('-' * 60)
    print('Adam with learning rate = 0.0')
    print('Min Loss: {}'.format(np.min(loss_00001)))
    print('Min Validation Loss: {}'.format(np.min(val_loss_00001)))

    axs[2].plot(loss_000001)
    axs[2].plot(val_loss_000001)
    axs[2].legend(["Training Loss", "Validation Loss"],loc="upper right")
    axs[2].set_title('Adam with learning rate = 0.00001')

    print('-' * 60)
    print('Adam with learning rate = 0.00001')
    print('Min Loss: {}'.format(np.min(loss_000001)))
    print('Min Validation Loss: {}'.format(np.min(val_loss_000001)))

    plt.show()

# Plot training data distribution (Track 1)
def plot_train_data_distribution_1():
    folder_names = [
        'track_1_normal', 'track_1_reverse',
        'track_1_normal_left_recovery_015', 'track_1_normal_right_recovery_015',
        'track_1_reverse_left_recovery_015', 'track_1_reverse_right_recovery_015']

    # Prepare track 1 data
    for index, folder_name in enumerate(folder_names):
        if index == 0:
            samples = readFromFile(folder_name)
        else:
            temp_samples = readFromFile(folder_name)
            samples = np.concatenate((samples, temp_samples), axis=0)

    multiple_camera_correction = 0.15

    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)
    steerings_include_flip = np.concatenate((steerings, -steerings), axis=0)
    shuffled_steerings = steerings_include_flip

    # Balancing data for Track 1
    count_02 = (0.2 > np.abs(shuffled_steerings)).sum()
    count_04 = ((np.abs(shuffled_steerings) >= 0.2)).sum()
    min_count = min(count_02, count_04)
    filtered_steerings = shuffled_steerings[0.2 > np.abs(shuffled_steerings)][:min_count]
    filtered_steerings = np.concatenate((filtered_steerings, shuffled_steerings[(np.abs(shuffled_steerings) >= 0.2)][:min_count]), axis=0)

    # Plot data
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 5))
    fig.suptitle('Steering Distribution (Tracked 1)')

    print('-' * 60)
    print('Steerings')
    print('Number of Samples: {}'.format(steerings.shape[0]))
    axs[0].hist(steerings, bins=11)
    axs[0].set_title('Raw Data')

    print('-' * 60)
    print('Steerings Including Flip')
    print('Number of Samples: {}'.format(steerings_include_flip.shape[0]))
    axs[1].hist(steerings_include_flip, bins=11)
    axs[1].set_title('Raw Data Including Flipped Image')

    print('-' * 60)
    print('Filtered Steerings')
    print('Number of Samples: {}'.format(filtered_steerings.shape[0]))
    axs[2].hist(filtered_steerings, bins=11)
    axs[2].set_title('Filtered Data')

    plt.show()

# Plot training data distribution (Track 1 & Track 2)
def plot_train_data_distribution_2():
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
            samples = readFromFile(folder_name)
        else:
            temp_samples = readFromFile(folder_name)
            samples = np.concatenate((samples, temp_samples), axis=0)

    multiple_camera_correction = 0.15
    steerings = np.array(samples[:,3], np.float)
    steerings = np.concatenate((steerings, steerings + multiple_camera_correction, steerings - multiple_camera_correction), axis=0)
    steerings_include_flip = np.concatenate((steerings, -steerings), axis=0)
    shuffled_steerings = steerings_include_flip

    # Balancing data for Track 1 and Track 2
    count_02 = (0.2 > np.abs(shuffled_steerings)).sum()
    count_04 = ((0.4 > np.abs(shuffled_steerings)) & (np.abs(shuffled_steerings) >= 0.2)).sum()
    count_06 = ((0.6 > np.abs(shuffled_steerings)) & (np.abs(shuffled_steerings) >= 0.4)).sum()
    count_08 = ((0.8 > np.abs(shuffled_steerings)) & (np.abs(shuffled_steerings) >= 0.6)).sum()
    count_10 = (np.abs(shuffled_steerings) >= 0.8).sum()
    min_count = min(count_02, count_04, count_06, count_08, count_10)
    filtered_steerings = shuffled_steerings[0.2 > np.abs(shuffled_steerings)][:min_count]
    filtered_steerings = np.concatenate((filtered_steerings, shuffled_steerings[(0.4 > np.abs(shuffled_steerings)) & (np.abs(shuffled_steerings) >= 0.2)][:min_count]), axis=0)
    filtered_steerings = np.concatenate((filtered_steerings, shuffled_steerings[(0.6 > np.abs(shuffled_steerings)) & (np.abs(shuffled_steerings) >= 0.4)][:min_count]), axis=0)
    filtered_steerings = np.concatenate((filtered_steerings, shuffled_steerings[(0.8 > np.abs(shuffled_steerings)) & (np.abs(shuffled_steerings) >= 0.6)][:min_count]), axis=0)
    filtered_steerings = np.concatenate((filtered_steerings, shuffled_steerings[np.abs(shuffled_steerings) >= 0.8][:min_count]), axis=0)

    # Plot data
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 5))
    fig.suptitle('Steering Distribution (Tracked 1 and 2)')

    print('-' * 60)
    print('Steerings')
    print('Number of Samples: {}'.format(steerings.shape[0]))
    axs[0].hist(steerings, bins=11)
    axs[0].set_title('Raw Data')

    print('-' * 60)
    print('Steerings Including Flip')
    print('Number of Samples: {}'.format(steerings_include_flip.shape[0]))
    axs[1].hist(steerings_include_flip, bins=11)
    axs[1].set_title('Raw Data Including Flipped Image')

    print('-' * 60)
    print('Filtered Steerings')
    print('Number of Samples: {}'.format(filtered_steerings.shape[0]))
    axs[2].hist(filtered_steerings, bins=11)
    axs[2].set_title('Filtered Data')

    plt.show()

# Plot model Loss for the training
def plot_model_loss(file_name):
    loss, val_loss = load_history(file_name)

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(["Training Loss", "Validation Loss"],loc="upper right")
    plt.title('Model Loss')

    plt.show()

# Plot the model loss against learning rate
plot_model_loss_against_learning_rate()

# Plot training data distribution (Track 1)
plot_train_data_distribution_1()
# Plot training data distribution (Track 1 & Track 2)
plot_train_data_distribution_2()

# Plot model Loss for the training (Track 1)
plot_model_loss('model_track_1')
# Plot model Loss for the training (Track 1 & Track 2)
plot_model_loss('model')