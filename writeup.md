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

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Since I take the start point to LeNet-5, entire model is similar to it. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					| Activation				|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten          | outputs 400   |
| Fully connected		| Flatten, outputs 120    |
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Speed Limit (30Km/h)][image4] ![Right-of-way at the next intersection][image5] ![Priority road][image6] 
![Stop][image7] ![Slippery road][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


