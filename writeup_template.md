# **Traffic Sign Recognition** 

## Writeup



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[training]: ./examples/training.png "training"
[valid]: ./examples/valid.png "valid"
[test]: ./examples/test.png "test"
[Grayscaling]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/11.png "Traffic Sign 1"
[image5]: ./examples/12.png "Traffic Sign 2"
[image6]: ./examples/13.png "Traffic Sign 3"
[image7]: ./examples/14.png "Traffic Sign 4"
[image8]: ./examples/15.png "Traffic Sign 5"

## [Rubric points](https://review.udacity.com/#!/rubrics/481/view)
###The following part will show the solution for every single point in Rubic Points.
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code](https://github.com/YimengZhu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed to all labels in form of a histodiagram. There are there histogram with respect to training set, cross validation set and test set. Furthermore I also plotted one traffic sign image in each kind of class.

![alt text][training]
![alt text][valid]
![alt text][test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As a first step, I decided to convert the images to grayscale because it can reduced the number of parameters of the input layer in the neural network without hurting the recognition of the sign.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][Grayscaling]

As a last step, I normalized the image data because it can accelerate the training of the neural network, and make the neural network not sensitive to some specific features.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaling image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU with Dropout					|												|
| Max pooling  with Dropout	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	        |1x1 stride, valid padding, outputs 10x10x16 			|
| RELU with Dropout					|												|
| Max pooling  with Dropout	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| input 120, output 84        									|
| RELU with Dropout					|												|
| Fully connected		| input 84, output 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer from tensorflow library. I set the learning rate to 0.001 and used 30 epoch to train the model. In every iteration I choosed a batch of training data with size of 128 after randomly shuffling the training set. The dropout problitiy was set to 0.8 to prevent overfitting. After each epoch, ther loss and accuracy of the validation set will be measured by the evalutate function. This pipline and evaluation approach I refered from [CarND-LeNet-Lab](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 94.9%
* test set accuracy of 94.7%.

If a well known architecture was chosen:
* What architecture was chosen: The LeNet from the previous lecture.
* Why did you believe it would be relevant to the traffic sign application:
The original purpose of this network is to recognize the handwritten digits in 32*32 pixel image. This task is very similar to the traffic sign recognition.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well:
The accuracy of the model are respectively 99.6%, 94.9% and 94.7% to the training, validation and test set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The forth image might be difficult to classify because it has many similar features to other signs like shape, colour etc.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-Way at next intersection|Right-of-Way at next intersection| 
| Priority road| Priority road|
| Yield| Yield|
| Stop| Stop|
| No vehicles| No vehicles|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is absolutly sure that this is a right-of-Way at next intersection (probability of almost 100%), and the image does contain a right-of-Way at next intersection. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00| Right-of-way at next intersection| 
| 0| Beware of ice/snow|
| 0| Children crossing|
| 0| Double curve |
| 0| Slippery road|


For the secound image, the model is relatively sure that this is a priority road (probability of 0.9824), and the image does contain a  priority road. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9824| Priority road| 
| .0068| Speed limit(30km/h)|
| .0030| No vehicles|
| .0025| Speed limit(50km/h)|
| .0011| Speed limit(80km/h)|

For the third image, the model is absolutly sure that this is a yield sign (probability of almost 1.00), and the image does contain a yield. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000| Yield									| 
| 0.0000| Ahead only|
| 0.0000| Priority road|
| 0.0000| Speed limit(30km/h)|
| 0.0000| Turn left ahead|

For the fourth image, the model is relatively sure that this is a stop (probability of 0.3338), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.3338| Stop sign| 
| 0.2667| Speed limit(30km/h)|
| 0.2458| Speed limit(80km/h)|
| 0.0710| Road work|
| 0.0208| Keep right|

For the fifth image, the model is relatively sure that this is a no vehicles (probability of 0.9849), and the image does contain a no vehicles sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9849| No vehicles| 
| 0.0041| Speed limit(50km/h)|
| 0.0020| Speed limit(70km/h)|
| 0.0020| Priority road|
| 0.0016| Speed limit(30km/h)|
