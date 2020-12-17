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
