# Deep Learning
Deep Learning is a subfiled of machine learning concerned with algorithms inspired by the structure and fuction of the brain called artifical neural networks. Learning an be supervised, semi-supervised or unsupervised. 

# TensorFlow
Created by the Google Brain team, TensorFlow is a free and open-source software library for numerical computation and large-scale machine learning. TensorFlow bundles together a slew of machine learning and deep learning (a.k.a neural networking) models and algorithms and makes them useful by way of a common metaphor. 

## Tensor
In mathematics, a tensor is an algebraic object that describes a relationship between sets of algebraic objects related to a vector space. Objects that tensors may map between include vectors and scalars, and even other tensors. 

# OpenCV
image and video analysis, like facial recognition and detection, license plate reading, photo editing, advanced robotic vision, and a whole lot more using Python.

# Anaconda Version Change
https://bblib.net/entry/%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-python-%EB%B2%84%EC%A0%84-%EB%B3%80%EA%B2%BD-%ED%95%98%EB%8A%94-%EB%B2%95

# Lecture
Classes by Sung Kim

Youtube Link : https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=1

Slide Link : https://drive.google.com/drive/folders/0B41Zbb4c8HVyMHlSQlVFWWphNXc

## 02 Linear Regression
### Hypothesis and Cost
$H(x) = Wx + b$


$cost(W, b) = \frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)})-y^{(i))})^{2}$


When you make a cost function, it has to be convex function. 

### Gradient Descent Algorithm
* Minimize ccost function
* Gradient descent is used many minimization problems
* For a give cost function, cost(W, b), it will find W, b to minisize cost
* It can be applied to more general function: cost(w1, w2, ...)

## 05 Logistic Classification
### Logistic Function (Sigmoid Function)
curved in two directions, like the letter "S", or the Greek c(sigma). The result value is between 0 and 1. 

## 06 Softmax Classification
The Softmax function takes as input a vector z of K real numbers, and normalized it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers. 

### One Hot-Encoding
One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. 
In the "colour" variable example, there are 3 categories and therefore 3 binary variables are needed. A "1" value is placed in the binary variable for the colour and "0" values for the other colours. For example:
|red|green|blue|
|---|---|---|
|1|0|0|
|0|1|0|
|0|0|1|


### Categorical Data
Categorical data are variables that contain label values rather thatn numeric values. Categorical variables are often called nominal.
Some exapmes include:
* A "pet" variables with the values: "dog" and "cat".
* A "colour" variables with the values: "red", "green" and "blue".
* A "place" variables with the values: "first", "second", and "third".

Each value represents a different category. Some categories may have a natural relationship to each other, such as a natural ordering. The "place" variable above does have a natural ordering of values. This type of categorical variable is called an ordinal variable. 

## 07 Application & Tips
### Learning Rate
Large learning rate cause overshooting problem. On the other hand, small learning rate needs many iterations until convergence and trapping in local minima.

### Data Preprocessing

### Overfitting

Solutions for overfitting:
* More training data
* Reduce the number of features
* Regularization

### Online Learning
### Training epoch/batch
In the neural network terminology:
  * one epoch = one forward pass and one backward pass of all the training examples.
  * batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
  * number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
  
Example: if you have 1000 training examples, and your batch sisze is 500, then it will take 2 iterations to complete 1 epoch. 

## 09 How to Solve XOR in Deep Learning
### 5 Steps of Using TensorBoard
1. From TF graph, decide which tensors you want to log
```
w2_hist = tf.summary.histogram("weight2", W2)
cost_summ = tf.summary.scalar("cost", cost)
```

2. Merge all summaries
```
summary = tf.summary.merge_all()
```

3. Create Writer and add graph
```
# Create summary writer
writer = tf.summary.FileWriter('./logs')  # set the file location
writer.add_graph(sess.graph)
```

4. Run summary merge and add_summary
```
s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
writer.add_summary(s, global_step=global_step)
```

5. Launch TensorBoard
```
tensorboard --logdir=./logs
```
```
# remote server, you can navigate to http://127.0.0.1:7007
local> $ ssh -L local_port:127.0.0.1:remote_port username@server.com
server> $ tensorboard -logdir=./logs/xor_logs
```

## 10 ReLU (Rectified Linear Unit)

### Geoffrey Hinton's Summary of Finidings up to Today
* Our labeled datasets were thousands of times too small.
* Our computers were milions of times too slow.
* We initialized the weights in a stupid way.
* We used the wrong type of non-linearity.

### Backpropagation (Chain Rule)

### Vanishing Gradient Problem 
We used the wrong type of non-linearity. Using sigmoid function make values converge into zero.

### Sigmoid
```
tf.sigmoid(tf.matmul(X, W) + b))
```

### ReLU (Rectified Linear Unit)
```
tf.nn.relu(tf.matmul(X, W) + b))
```

### Leaky ReLU

### ELU

### Regularization: Dropout
randomly set some neurons to zero in the forward pass. It forces the network to have a redundant representation.
```
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W), b))
L1 = tf.nn.dropout(_L1, dropout_rate)

# and when you train the data, use dropout but for the evaluation, you need to use all data from datasets. 
```

### Xavier Initialization
```
W = tf.get_variable("W", shape=[784, 256], initializer=tf.contrib.laters.xavier.initilizer())
```

## 11 CNN (Convolutional Neural Network)

* Stride
* Filter(F)
* N : total image size (length/width)
* Output size : (N - F) / stride + 1
* In Practice: Common to zero pad the border, in general, common to see CONV layers with stride 1, filters of size F * F, and zero-padding with (F - 1) / 2. (will preserve size spatially)
e.g. F = 3 => zero pad with 1
     F = 5 => zero pad with 2
     F = 7 => zero pad with 3


