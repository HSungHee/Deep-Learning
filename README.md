# Deep Learning
Deep Learning is a subfiled of machine learning concerned with algorithms inspired by the structure and fuction of the brain called artifical neural networks. Learning an be supervised, semi-supervised or unsupervised. 

# Deep Network
* Take a long time for training
  - Many forward/backward propagation and weight updates
  - Many metrics multiplications
  
* Very quick for testing and use in practice
  - One simple forward propagation

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

## 02. Linear Regression
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
Data preprocessing is an important step in the data mining process. If there is much irrelevant and redundant information present or noisy and unreliable data, then knowledge discovery during the training phase is more difficult. Data preparation and filtering steps can take considerable amount of processing time. Data preprocessing includes cleaning, Instance selection, normalization, transformation, feature extraction, and selection, etc. The product of data preprocessing is the final training set. 

Tasks of data preprocessing:
* Data cleansing
* Data editing
* Data reduction
* Data wrangling

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
### 5 Steps of Using Tensor

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

## 10. ReLU (Rectified Linear Unit)

### TensorBoard
https://webnautes.tistory.com/1288

### Geoffrey Hinton's Summary of Finidings up to Today
* Our labeled datasets were thousands of times too small.
* Our computers were milions of times too slow.
* We initialized the weights in a stupid way.
* We used the wrong type of non-linearity.

### Backpropagation (Chain Rule)
Backpropagation, short for "backward propagation of errors", is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error funtion, the method calculates the gradient of the rror function with respect to the neural network's weights. 

### Vanishing Gradient Problem 
We used the wrong type of non-linearity. Using sigmoid function make values converge into zero. In machine learning, the vanishing gradient problem is encountered when training artificial neural networks with gradient-based learning methods and backpropagation. In such methods, each of the neural network's weights receives an update proportional to the partial derivative of the rror function with respect to the current weight in each iteration of training. The problem is that in some cases, the gradient will be vanishingly small, effectively preventing the weight from changing its value. In the worst case, this may completely stop the neural network from further training.

### Sigmoid
A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. A common example of a sigmoid function is the logistic function shown in the first figure and defined by the formula. 

$$ S(x) = 1/(1 + e^{-x}) = e^x/(e^x + 1) $$

```
tf.sigmoid(tf.matmul(X, W) + b))
```

### ReLU (Rectified Linear Unit)
The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural netowrks because a model that uses it is easier to train and often achieves beter performance. 
```
tf.nn.relu(tf.matmul(X, W) + b))
```

### Leaky ReLU
The Leaky ReLU (LReLU or LReL) modifies the function to allow small negative values when the input is less than zero.

### Regularization: Dropout
Randomly set some neurons to zero in the forward pass. It forces the network to have a redundant representation.
```
dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X, W), b))
L1 = tf.nn.dropout(_L1, dropout_rate)

# and when you train the data, use dropout but for the evaluation, you need to use all data from datasets. 
```

### Xavier Initialization
Xavier initialization, originally proposed by Xavier Glorot and Yoshua Bengio in "Understanding the difficulty of training deep feedforward neural networks", is the weights initialization technique that tries to make the variance of the outputs of a layer to be equal to the variance of its inputs.
```
W = tf.get_variable("W", shape=[784, 256], initializer=tf.contrib.laters.xavier.initilizer())
```

## 11. CNN (Convolutional Neural Network)
In deep learning, a convolutional neural netowrk (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fullly connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "fully-connectedness" of theses networks makes them prone to overfiitting data. Typical ways of regularization include adding some form of magnitude measurement of weights to the loss function. CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme.

* Stride
* Filter(F)
* N : total image size (length/width)
* Output size : (N - F) / stride + 1
* In Practice: Common to zero pad the border, in general, common to see CONV layers with stride 1, filters of size F * F, and zero-padding with (F - 1) / 2. (will preserve size spatially)

     e.g. F = 3 => zero pad with 1

     F = 5 => zero pad with 2

     F = 7 => zero pad with 3

### Case Study: LeNet-5
LeNet is a convolutional neural network structure proposed by Yann LeCun et al. in 1989. In general, LeNet refers to lenet-5 and is a simple convolutional neural network. Convolutional neural networks are a kind of feed-forward neural network whose artificial neurons can respond to a part of the surrounding cells in the coverage range and perform well in large-scale image processing.

### Case Study: AlexNet
AlexNet is the name of a convolutional neural network (CNN), designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor.

AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) during training.

### Case Study: ResNet
A residual neural network (ResNet) is an artificial neural network (ANN) of a kind that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers. Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in between. An additional weight matrix may be used to learn the skip weights; these models are known as HighwayNets. Models with several parallel skips are referred to as DenseNets. In the context of residual neural networks, a non-residual network may be described as a plain network.

## RNN (Recurrent Neural Network)
A recurrent neural netowrk (RNN) is a class of artificial neural networks where connections between nodes from a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of input. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. 

We can process a sequence of vectors x by applying a recurrence formula at every time step:

$$ h_t = f_w(h_{t-1}, x_t) $$

Notice: the same function and the same set of parameters are used at every time step.

Applications:
* Language Modeling
* Speech Recognition
* Machine Translation
* Conversation Modeling/Question Answering
* Image/Video Captioning
* Image/Music/Dance Generation

Several advanced models:
* Long Short Term Memory (LSTM)
* GRU by Cho et al. 2014

### Sequence Data
Sequential pattern mining is a topic of data mining concerned with finding statistically relevant patterns between data examples where the values are delivered in a sequence. It is ususally presumed that the values are discrete, and thus time series mining is closely related, but ususally considered a different activity. 

* We don't understand one word only
* We understand based on the previous words + this word (time series)
* NN/CNN cannot do this

## 13. Tensorflow on AWS

### GPU
A graphics processing unit(GPU), also occasionally called visual processing unit (VPU), is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display. 

```
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below
$ export TF_BINARY_URL=http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Python 2
$ sudo pip install --upgrade $TF_BINARY_URL
```

### Add Path
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/urs/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Reference

- Definition 
* https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.
* https://www.wikipedia.org/
* https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
* https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize
