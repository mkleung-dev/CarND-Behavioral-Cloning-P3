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
[track_1_video_beginning]: ./image/track_1_video_beginning.jpg "Track 1 Video Beginning"
[track_2_video_beginning]: ./image/track_2_video_beginning.jpg "Track 2 Video Beginning"

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
|model.h5|Containing a trained convolution neural network for track 1 and track 2.|
|writeup.md|Summarizing the results.|
|plot.py|Containing the script to plot result.|
|video.mp4|Video recording of my vehicle driving autonomously around the track 1.|
|video_track_2.mp4|Video recording of my vehicle driving autonomously around the track 2.|

### 2. Code Quality

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model.
The plot.py file contains the code for ploting the result.
The drive.py file contains the code for computing the throttle power and the steering value by processing the image from the simulator.
They contain comments to explain how the code works.

### 3. Model Architecture and Training Strategy

#### 1. Appropriate model architecture

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 32 (behavioral_cloning_model() in model.py lines 250-279).
The input RGB image is first normalized in the range of -1 and 1 using a Keras lambda layer.
The image is then cropped so that it only focus on the road.
The image is then fed to the convolution neural network.
The model includes RELU layers to introduce nonlinearity.
The final output is steering angle without activation layer.
Detail is illustrated in the next section. 

#### 2. Reducing overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 273,276). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 308,358).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. (model.py lines 303,353). 
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
The convolution neural network consisted of several 3 x 3 convolution kernels and fully connected network.
I believed that it could work because the convolution neural can work well with images.

After each convolution layer and dense layer of the hidden layers, RELU activation is used to make the network non-linear.
To combat the overfitting, for each output of hiden dense layer, drop out of 80% passing is used.

I also tried to implement the neutral network introduced in (https://developer.nvidia.com/blog/deep-learning-self-driving-cars/).
I found that the performance of my network is similar to the network introduced by Nvidia. 

Then I run the simulator to see how well the car was driving around track 1.
However, it was found that the car would move outside of the track in some sharp turn.
At the same time, I discovered that there were much more data with very small steering value than data with large steering value.
I tried to randomly remove some of data with small steering values in each epoch to train the model.
After balancing the data, the car could always run perfectly around track 1.

After that I tried to run the simular to see how well the car was driving around track 2.
It moved outside of the lane easily.
Therefore, I collect more data from track 2 and trained the model using both track.
It still moved outside of the lane easily.
I tried to get more data from those difficult positions.
At the same time, I discovered that the sample distribution of all data from track 1 and track 2 was different.
This time, I tried to balance the data by randomly remove some data so that there are roughly the same number of samples for different steering values.

At the end of the process, the vehicle is able to drive autonomously around the track 1 and the track 2 without leaving the road using a single model.

#### 2. Model Architecture

The final model architecture (behavioral_cloning_model() in model.py lines 250-279) consists of a convolution neural network with the following layers and layer sizes.

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

Here is the visualization of the architecture.

![cnn]

#### 3. Training Set Creation & Training Process

**For Track 1** 

To capture driving behavior, 2 laps on track 1 are recorded using center lane driving.

Here is the example of images from left, center, and right cameras during center lane driving:

|Left camera|Center camera|Right camera|
|:---|:-----:|:-----------------:|
|![track_1_normal_left_camera]|![track_1_normal_center_camera]|![track_1_normal_right_camera]|

The steering values for left camera and right camera were compensated by 3.75 deg so the car could return to the center of the lane.

Two laps on track 1 are also recorded using left lane driving and right lane driving respectively.
They are used for recovery driving. Additional 3.75 deg steering are added so that the car can return to the center of the lane.

Here is the example of images from left, center, and right cameras during left lane driving:

|Left camera|Center camera|Right camera|
|:---|:-----:|:-----------------:|
|![track_1_normal_left_camera_left]|![track_1_normal_center_camera_left]|![track_1_normal_right_camera_left]|

Here is the example of images from left, center, and right cameras during right lane driving:

|Left camera|Center camera|Right camera|
|:---|:-----:|:-----------------:|
|![track_1_normal_left_camera_right]|![track_1_normal_center_camera_right]|![track_1_normal_right_camera_right]|

All the above collection processes were repeated when driving the track in opposition direction.
To augment the data set, all images were flipped to duplicate the number of samples.
After the collection process of the track 1, the following table shows the number of samples.

However, there was bias in the sample data. There were much more data with small steering values than data with large steering values.
Therefore, when training for each epoch, some data with small steering values was removed randomly.
The following histogram shows the distribution of the steering values before and after the filtering.

![track_1_steering_histogram]

The following table shows the number of samples used in track 1.

|Item|Track 1|
|:---|:-----:|
|Raw data|136386|
|Raw data including flips|272772|
|Filtered data|185768|

30 % of the data was put into the validation set.
The validation set helped determine if the model was over or under fitting.

The model was trained using Adam Optimizer with the following parameters.
The training is implemented in (train_model_for_track_1() in model.py lines 282-318).
|Item            |Value |
|:---------------|:----:|
|Batch size      |32    |
|Number of epochs|10    |
|Learning rate   |0.0001|

![epoch_study_1]

The model by this method could enable the car to run around track 1, but not track 2.

**For Track 1 and Track 2** 

In track 2, there are 2 lanes. For each lane, 2 laps of data in both directions are recorded.
The following table shows some of the center images.

|Left lane|Right lane|Left lane in opposition direction|Right lane in opposition direction|
|:-------:|:--------:|:-------------------------------:|:--------------------------------:|
|![track_2_normal_center_camera_left_lane]|![track_2_normal_center_camera_right_lane]|![track_2_reverse_center_camera_left_lane]|![track_2_reverse_center_camera_right_lane]|

In addition, I found that there were more lighting variations when starting the simulator in 'fantasy' mode.
There are more shadows on the road.
For each lane, 2 laps of data in both directions are also recorded in 'fantasy' mode.
The following table shows some of the center images in 'fantasy' mode.

|Left lane|Right lane|Left lane in opposition direction|Right lane in opposition direction|
|:-------:|:--------:|:-------------------------------:|:--------------------------------:|
|![track_2_normal_center_camera_left_lane_fantasy]|![track_2_normal_center_camera_right_lane_fantasy]|![track_2_reverse_center_camera_left_lane_fantasy]|![track_2_reverse_center_camera_right_lane_fantasy]|

However, when I tested the model trained using the above model, the car still could not stay in the center of the lane in some positions.
Therefore, get more sample data in those position.
The following table shows some of the center images for this case.

|Difficult case 1|Difficult case 2|Difficult case 3|Difficult case 4|
|:-------:|:--------:|:-------------------------------:|:--------------------------------:|
|![track_2_special_1]|![track_2_special_2]|![track_2_special_3]|![track_2_special_4]|

However, there was bias in the sample data.
There were more data with medium steering values than data with other steering values.
Therefore, when trainig for each epoch, randomly removed some data so that there were roughly the same number of samples for different steering values.
The following histogram showed the distribution of the steering values before and after the filtering.

![track_1_and_track_2_steering_histogram]

The following table shows the number of samples used in both track 1 and track 2.

|Item|Track 1|Track 1 and Track 2|
|:---|:-----:|:-----------------:|
|Raw data|136386|504918|
|Raw data including flips|272772|1009836|
|Filtered data|185768|165180|

30 % of the data was put into the validation set.
The validation set helped determine if the model was over or under fitting.

The model was trained using Adam Optimizer with the following parameters.
The training is implemented in (train_model_for_track_1_and_track_2() in model.py lines 321-368).

|Item            |Value |
|:---------------|:----:|
|Batch size      |32    |
|Number of epochs|20    |
|Learning rate   |0.0001|

![epoch_study_2]

The model by this method could enable the car to run around both track 1 and track 2 perfectly.

### 5. Simulation

Here is the video recording of my vehicle driving autonomously around the track 1.
[![track_1_video_beginning]](./video/video_track_1.mp4)

Here is the video recording of my vehicle driving autonomously around the track 2.
[![track_2_video_beginning]](./video/video_track_2.mp4)

Occasionally, the car may stall around track.
To handle this, give a negative throttle when stalling is detected (drive.py line 41-51).
