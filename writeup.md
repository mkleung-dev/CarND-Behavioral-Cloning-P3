# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track 1 without leaving the road
* Test that the model successfully drives around track 2 without leaving the road
* Summarize the results with a written report

[//]: # (Image References)\

[epoch_study_1]: ./image/epoch_study_1.png "Epoch Study 1"
[epoch_study_2]: ./image/epoch_study_2.png "Epoch Study 2"
[learning_rate_study]: ./image/learning_rate_study.png "Learning Rate Study"
[cnn]: ./image/cnn.png "Convolution Neurual Network"
[track_1_steering_histogram]: ./image/track_1_steering_histogram.png "Track 1 Steering Histogram"
[track_1_and_track_2_steering_histogram]: ./image/track_1_and_track_2_steering_histogram.png "Track 1 and Track 2 Steering Histogram"

[track_1_normal_left_camera]: ./image/track_1_normal_left_camera.jpg "Track 1 Left Camera"
[track_1_normal_center_camera]: ./image/track_1_normal_center_camera.jpg "Track 1 Center Camera"
[track_1_normal_right_camera]: ./image/track_1_normal_right_camera.jpg "Track 1 Right Camera"
[track_1_normal_left_camera_left]: ./image/track_1_normal_left_camera_left.jpg "Track 1 Left Camera Recovery from Left"
[track_1_normal_center_camera_left]: ./image/track_1_normal_center_camera_left.jpg "Track 1 Center Camera Recovery from Left"
[track_1_normal_right_camera_left]: ./image/track_1_normal_right_camera_left.jpg "Track 1 Right Camera Recovery from Left"
[track_1_normal_left_camera_right]: ./image/track_1_normal_left_camera_right.jpg "Track 1 Left Camera Recovery from Right"
[track_1_normal_center_camera_right]: ./image/track_1_normal_center_camera_right.jpg "Track 1 Center Camera Recovery from Right"
[track_1_normal_right_camera_right]: ./image/track_1_normal_right_camera_right.jpg "Track 1 Right Camera Recovery from Rogjt"
[track_2_normal_center_camera_left_lane]: ./image/track_2_normal_center_camera_left_lane.jpg "Track 1 Center Camera on Left Lane"
[track_2_reverse_center_camera_left_lane]: ./image/track_2_reverse_center_camera_left_lane.jpg "Track 1 Center Camera on Left Lane in Opposite Direction"
[track_2_normal_center_camera_right_lane]: ./image/track_2_normal_center_camera_right_lane.jpg "Track 1 Center Camera on Right Lane"
[track_2_reverse_center_camera_right_lane]: ./image/track_2_reverse_center_camera_right_lane.jpg "Track 1 Center Camera on Right Lane in Opposite Direction"
[track_2_normal_center_camera_left_lane_fantasy]: ./image/track_2_normal_center_camera_left_lane_fantasy.jpg "Track 1 Center Camera on Left Lane in Fantasy Mode"
[track_2_normal_center_camera_right_lane_fantasy]: ./image/track_2_normal_center_camera_right_lane_fantasy.jpg "Track 1 Center Camera on Left Lane in Opposite Direction in Fantasy Mode"
[track_2_reverse_center_camera_left_lane_fantasy]: ./image/track_2_reverse_center_camera_left_lane_fantasy.jpg "Track 1 Center Camera on Right Lane in Fantasy Mode"
[track_2_reverse_center_camera_right_lane_fantasy]: ./image/track_2_reverse_center_camera_right_lane_fantasy.jpg "Track 1 Center Camera on Right Lane in Opposite Direction in Fantasy Mode"
[track_2_special_1]: ./image/track_2_special_1.jpg "Track 1 and Track 2 Steering Histogram"
[track_2_special_2]: ./image/track_2_special_2.jpg "Track 1 and Track 2 Steering Histogram"
[track_2_special_3]: ./image/track_2_special_3.jpg "Track 1 and Track 2 Steering Histogram"
[track_2_special_4]: ./image/track_2_special_4.jpg "Track 1 and Track 2 Steering Histogram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### 1. Files Submitted

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
|File|Description|
|:--:|:----------|
|model.py|Containing the script to create and train the model.|
|drive.py|Containing the script to drive the car in autonomous mode.|
|model.h5|Containing a trained convolution neural network.|
|writeup.md|Summarizing the results.|

### 2. Code Quality

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model.
It contains comments to explain how the code works.

### 3. Model Architecture and Training Strategy

#### 1. Appropriate model architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 32 (model.py lines 18-24).
The input RGB image is first normalized in the range of -1 and 1 using a Keras lambda layer.
The image is then cropped so that it only focus on the road.
The image is then fed to the convolution neural network.
The model includes RELU layers to introduce nonlinearity.
The final output is steering angle without activation layer.
Detail is illustrated in the next section. 

#### 2. Reducing overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer.
I tried to trained the model with smaller data to study how the learning rate affected the result.

![learning_rate_study]

Learning rate is set to 0.0001 as it allowed the model to achieve lowest loss value (0.034).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
For track 1, I tried the following ways to get training data in the video setting of both fastest and fantasy.

1. Center lane driving (Images from center, left, and right cameras)
2. Left side driving (Images from center, left, and right cameras)
3. Right side driving (Images from center, left, and right cameras)

For track 2, I tried the following ways to get training data in the video setting of both fastest and fantasy.

1. Center lane driving (Images from center, left, and right cameras)

More details would be describted the next section. 

### 4. Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to trying different convolution neural network.

My first step was to use a convolution neural network model similar to the VGG.
I made use of many 3 x 3 convolution kernel and then fully connected network to construct the model.
I believe that it can work because the convolution neural can work well with 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

Then I run the simulator to see how well the car was driving around track 1.
However, it was found that the car would move outside of the track in some sharp turn.
At the same time, I discovered that there were much more data with very small steering value than data with large steering value.
I tried to randomly remove some of data with small steering values in each epoch to train the model.
After balancing the data, the car can run always perfectly around track 1.

After that I tried to run the simular to see how well the car was driving around track 2.
It moved outside of the lane easily.
Therefore, I collect more data from track 2 and trained the model using both track.
It still moved outside of the lane easily.
I tried to get more data from those difficult positions.
At the same time, I discovered that the sample distribution of all data from track 1 and track 2 was different.
This time, I tried to balance the data by randomly remove some data so that there are roughly the same number of samples for different steering values.

At the end of the process, the vehicle is able to drive autonomously around the track 1 and the track 2 without leaving the road using a single model.

#### 2. Model Architecture

The final model architecture (model.py) consists of a convolution neural network with the following layers and layer sizes.

|Layer             |Description                 |Output       |Parameter|
|:-----------------|:---------------------------|:-----------:|:-------:|
|Input             |Color image                 |160 x 320 x 3|0        |
|Normalization     |Color image                 |160 x 320 x 3|0        |
|Cropping          |Color image                 |100 x 320 x 3|0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |98 x 318 x 16|448      |
|RELU              |Relu Activation             |98 x 318 x 16|0        |
|Max pooling 2 x 2 |2 x 2 stride                |49 x 159 x 16|0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |47 x 157 x 16|2320     |
|RELU              |Relu Activation             |47 x 157 x 16|0        |
|Max pooling 2 x 2 |2 x 2 stride                |23 x 78  x 16|0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |21 x 76 x 32 |4640     |
|RELU              |Relu Activation             |21 x 76 x 32 |0        |
|Max pooling 2 x 2 |2 x 2 stride                |10 x 38 x 32 |0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |8 x 36 x 32  |9248     |
|RELU              |Relu Activation             |8 x 36 x 32  |0        |
|Max pooling 2 x 2 |2 x 2 stride                |4 x 18 x 32  |0        |
|Convolution 3 x 3 |1 x 1 stride, valid padding |2 x 16 x 32  |9248     |
|RELU              |Relu Activation             |2 x 16 x 32  |0        |
|Max pooling 2 x 2 |2 x 2 stride                |1 x 8 x 32   |0        |
|Flatten           |Flatten                     |256          |0        |
|Dense             |Dense network               |1024         |263168   |
|RELU              |Relu Activation             |1024         |0        |
|Drop out          |70 % Passing                |1024         |0        |
|Dense             |Dense network               |512          |524800   |
|RELU              |Relu Activation             |512          |0        |
|Drop out          |70 % Passing                |512          |0        |
|Dense             |Dense network               |1            |513      |

Total parameters: 814,385

Here is a visualization of the architecture

![cnn]

#### 3. Training Set Creation & Training Process

**For Track 1** 

To capture driving behavior, two laps on track 1 are recorded using center lane driving. Here is an example of images from left, center, and right cameras during center lane driving:

|Left camera|Center camera|Right camera|
|:---|:-----:|:-----------------:|
|![track_1_normal_left_camera]|![track_1_normal_center_camera]|![track_1_normal_right_camera]|

The steering values for left camera and right camera are compensated by 3.75 deg so the car can return to the center of the lane.

Two laps on track 1 are also recorded using left lane driving and right lane driving respectively.
They are used for recovery driving. Additional 3.75 deg steering are added so that the car can return to the center of the lane.

Here is an example of images from left, center, and right cameras during left lane driving:

|Left camera|Center camera|Right camera|
|:---|:-----:|:-----------------:|
|![track_1_normal_left_camera_left]|![track_1_normal_center_camera_left]|![track_1_normal_right_camera_left]|

Here is an example of images from left, center, and right cameras during right lane driving:

|Left camera|Center camera|Right camera|
|:---|:-----:|:-----------------:|
|![track_1_normal_left_camera_right]|![track_1_normal_center_camera_right]|![track_1_normal_right_camera_right]|

All the above processes are repeated when driving the track in the opposition direction.
To augment the data set, all images are flipped to duplicate the number of samples.
After the collection process of the track 1, the following table show the number of data point.

However, there is bias in the sample data. There are much more data with small steering values than data with large steering values. Therefore, when trainig for each epoch, I randomly remove some data with small steering values. The following histogram show the distribution of the steering values before and after the filtering.

![track_1_steering_histogram]

The following table shows the number of samples used in track 1.

|Item|Track 1|
|:---|:-----:|
|Raw data|136386|
|Raw data including flips|272772|
|Filtered data|185768|

I finally put 30 % of the data into a validation set.
The validation set helped determine if the model was over or under fitting.

The model was trained using Adam Optimizer with the following parameters.
The training is implemented in (model.py ).
|Item            |Value |
|:---------------|:----:|
|Batch size      |32    |
|Number of epochs|20    |
|Learning rate   |0.0001|

![epoch_study_1]

The model by this method can enable the car to run perfect around track 1, but not track 2.

**For Track 1 and Track 2** 

In track 2, there are 2 lanes. For each lane, 2 laps of data in both directions are recorded.
The following table show some of the center images.

|Left lane|Right lane|Left lane in opposition direction|Right lane in opposition direction|
|:-------:|:--------:|:-------------------------------:|:--------------------------------:|
|![track_2_normal_center_camera_left_lane]|![track_2_normal_center_camera_right_lane]|![track_2_reverse_center_camera_left_lane]|![track_2_reverse_center_camera_right_lane]|

In addition, I found that there are more lighting variations when starting the simulator in 'fantasy' mode.
There are more shadows on the road.
For each lane, 2 laps of data in both directions are also records in 'fantasy' mode.
The following table show some of the center images in 'fantasy' mode.

|Left lane|Right lane|Left lane in opposition direction|Right lane in opposition direction|
|:-------:|:--------:|:-------------------------------:|:--------------------------------:|
|![track_2_normal_center_camera_left_lane_fantasy]|![track_2_normal_center_camera_right_lane_fantasy]|![track_2_reverse_center_camera_left_lane_fantasy]|![track_2_reverse_center_camera_right_lane_fantasy]|

However, when I test the model trained using the above model, the car still cannot drive will in some position.
The following table show some of the center images for this case.

|Difficult case 1|Difficult case 2|Difficult case 3|Difficult case 4|
|:-------:|:--------:|:-------------------------------:|:--------------------------------:|
|![track_2_special_1]|![track_2_special_2]|![track_2_special_3]|![track_2_special_4]|

However, there is bias in the sample data.
There are more data with medium steering values than data with other steering values.
Therefore, when trainig for each epoch, I randomly remove some data so that there are roughly the same number of samples for different steering values.
The following histogram show the distribution of the steering values before and after the filtering.

The following table show some of the center images.
![track_1_and_track_2_steering_histogram]

The following table shows the number of samples used in both track 1 and track 2.

|Item|Track 1|Track 1 and Track 2|
|:---|:-----:|:-----------------:|
|Raw data|136386|504918|
|Raw data including flips|272772|1009836|
|Filtered data|185768|165180|

I finally put 30 % of the data into a validation set. The validation set helped determine if the model was over or under fitting.

The model was trained using Adam Optimizer with the following parameters.
The training is implemented in (model.py).
|Item            |Value |
|:---------------|:----:|
|Batch size      |32    |
|Number of epochs|20    |
|Learning rate   |0.0001|

![epoch_study_2]

The model by this method can enable the car to run perfect around both track 1 and track 2.

### 5. Simulation

