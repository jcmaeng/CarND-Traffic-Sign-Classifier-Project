# Traffic Sign Recognition

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1-1]: ./writeup_images/training_examples.png "Visualization (training set)"
[image1-2]: ./writeup_images/validation_examples.png "Visualization (validation set)"
[image1-3]: ./writeup_images/testing_examples.png "Visualization (testing set)"
[image2]: ./writeup_images/valid_accuracy.png "Validation Set Accuracy"
[image4]: ./traffic-signs-imgs-test/1.jpg "Speed Limit (30Km/h)"
[image5]: ./traffic-signs-imgs-test/11.jpg "Right-of-way at the next intersection"
[image6]: ./traffic-signs-imgs-test/12.jpg "Priority road"
[image7]: ./traffic-signs-imgs-test/14.jpg "Stop"
[image8]: ./traffic-signs-imgs-test/23.jpg "Slippery road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/jcmaeng/CarND_Term1_P2_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Basic Summary
I used the numpy methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is bar charts showing how the data set is consisted.
These bar charts shows how many data are for each traffic sign.

![][image1-1] ![][image1-2] ![][image1-3]

### Design and Test a Model Architecture

At the preprocessing step, I try to apply two things; normalization and grayscaling. First, I apply normalization with the equation which is suggested in the guidelines of this project. The first normalization equation is shown below.

  (pixels - 128.0) / 128.0

However, with this equation, I cannot increase the accuracy over 0.90. In the mentoring, my mentor gave me some advice and he suggest another equation for normalization.

  (pixels / 255.0) - 0.5

Difference between two equation is whether it is zero-centered or not. I don't know which is better equation but I think that the second one fits better to my model.

At the grayscaling step, I convert data to grayscale but it does not helpful to increase accuracy. On the contrary, the accuracy goes down. After some trial for applying grayscale, I decide to exclude it in my project.

Since I take the start point to LeNet-5, entire model is similar to it. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|                 				|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten          | outputs 400   |
| Fully connected		| outputs 120    |
| RELU          |                         |
| Dropout       | keep_prop 0.6           |
| Fully connected		| outputs 84     |
| RELU          |                         |
| Dropout       | keep_prop 0.9           |
| Fully connected		| outputs 43     |
| Softmax				|       									|

To train the model, I used AdamOptimiser with learning rate, batch size and number of epochs as follows.

- learning rate: 0.0001~0.1, finally 0.001
- batch size: 32~512, finally 128
- epochs: 10~100, finally 100

I run the model with some range of parameter values then fix values with final one to achieve the accuracy goal.
Finally, I use evaluate() function which is the same function in LeNet-5 solution. I think that it is good enough to my model since my model is from LeNet-5.

As I mentioned previously, I try to run the model with various parameter values and then take the values for highest accuracy.

My final model results were:
* validation set accuracy: ~0.954 (refers the following graph)
![][image2]
* test set accuracy of: 0.939

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Speed Limit (30Km/h)][image4] ![Right-of-way at the next intersection][image5] ![Priority road][image6] 
![Stop][image7] ![Slippery road][image8]

The first image might be difficult to classify 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)        	| 
| Right-of-way at the next intersection		| Right-of-way at the next intersection |
| Priority road					| Priority road											|
| Stop	      		| Stop					 				|
| Slippery Road			| Slippery Road      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
For the new five images, the model predicts exact signs. The table below is shown the probabilities of the each image

| image #         	|     Prediction 1 (probability)		| Prediction 2 (probability)    |  Prediction 3 (probability)    |
|:---------------------:|:-----------------------------:|:-----------------------------:|:------------------------------:| 
| 1      		| Speed limit (30km/h)   (1.)							   | Double curve (9.06e-14)      | Right-of-way at the next intersection (2.90e-15)
| 2     	  | Right-of-way at the next intersection  (1.) | Slippery Road (6.17e-10)    | Pedestrians (3.63e-13)         | 
| 3			    | Priority road		(1.)									     | No entry (8.84e-14)          | Stop (8.45e-24)                |
| 4	      	| Stop					 	(1.)			                 | Bicycle crossing (5.30e-23) | Yield (3.11e-26) |
| 5			    | Slippery Road   (1.)   							       | Dangerous curve to the left (6.77e-23) |Right-of-way at the next intersection (2.89e-27) |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
I tried to visualize my model but I couldn't finish it.


