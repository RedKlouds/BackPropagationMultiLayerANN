# Multi-Layer Perceptron  

### Abstract

This project is made to help gain deeper understanding of how Artificial Neural Networks work. In this project we 
examine the performance differences between using a multi layered neural network, a Hebbiean learning network, and a single 
layered perceptron network. During this project I learned that with a mutlilayered perceptron, the input space is in deed split 
into multiple regions which allow this network to perform good classification of input data. The techniques namely Back-propagation
for minimizing the loss function of the approximator helps to steer our network towards the minimum of the error. As explored using 
back-propagation is a very efficent way to teach the network correct inputs.
 
---
### Introduction
In this project we use the following images which consist of 5X6 pixels of the digits 0 - 9. The task of this project is to implement 
a multi layer neural network from scratch. This network will allow deeper insight on the inner mathematical workings of the network, 
which will help us compare the performance measures of how well back-prorogation works.

Training set for this project.
![trainingset](https://image.ibb.co/ju1jKb/Sample_IMages.png)

---
\
### Architecture 
![Architecture](https://image.ibb.co/nr5HFG/Artchetectre.png)
In this implementation I will use a 3-layered perceptron design.
#### specifications (Hyper Parameters)

**Network Parameters**\
Learning Rate: Static, .10\
Neuron Count: 40\
Connections: 300

**First layer:**\
Activation Function: Log Sigmoid\
Neuron Count: 30\
Weights: random uniform\
bias: random uniform

**Second layer**\
Activation Function: Pure Linear\
Neuron Count: 10\
Weights: random int\
bias: random int


---
\
### Results



below are some resulting benchmarks to understand the networks behavior graphically\n
alpha .005 with 3 input patterns, over 400 epochs
![.005](https://image.ibb.co/jrYSJR/005_alpha_3_patterns_perf.png)
![.005](https://image.ibb.co/hMHnJR/005_alpha_10_patterns_perf.png)
The top image shows the mean squared error during training the network to recognize
TOP: 3 digits
BOTTOM: 10 digits
we can see that the mean squared error during training even with 400 epochs are nowhere near 50%
Below we now evaluate the performance of the network with some noise.
Noise was produced by making a premutated random index of pixels to change in each image, then running those
through the network and checking if they were correctly recognized.
![.005](https://image.ibb.co/eMiSJR/005_alpha_3_patterns_noise.png)
![.005](https://image.ibb.co/ibwQ56/005_alpha_10_patterns_noise.png)
TOP: is performance with 3 input samples trained
BOTTOM: performance with 10 input samples trained
As we can see it is expected that becasue the network was unable to properly train itself
to recognizing these input patterns the error of recognizing noisey/disfigured figures were poorly recognized.

---
Below we have images the learning rate moved up to .1, over 400 epochs with the same training samples as above
![.1](https://image.ibb.co/caWQ56/1_alpha_3_patterns_perf.png)
![.1](https://image.ibb.co/e4SbCm/1_alpha_10_patterns_perf.png)
TOP: training performance on .1 learning rate 3 Patterns
BOTTOM training performance on .1 learning rate 10 Patterns

As we can see and expect a much smoother curve of the mean squared error
coming down, to explain that the learning rate passed 20% error over the training set took about 100 epochs only
which tells us we have convergence of error
->Results are expected since the learning rate in a back-propagated algorithm tells the network how large of a step to take in the 
gradient decent in order to minimize this error/loss function, the larger the step in this case of the data sample the quicker can achieve a minimized
loss function for the network

![perfor](https://image.ibb.co/jxtuyR/1_alpha_10_patterns_noise.png)
now the interesting thing about this network is when we test the performance with degraded patterns, our error goes up starting at 4 changed pixels from the original
-> the complications could be due to a very large learning rate our algoirthm has learned or overfitted the training sample and can hardly
recognize noisey data, we can overcome this by applying generlization methods such as early stopping by creating a seperate validation dataset
we can compare when the validationset start to show activity to stop training the network.


gradient decent, is essentially what we are computing when we take the derivative of the transfer function per layer

*self pre-defined multi-layered neural network to understanding machine learning with ANN better

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
```
numpy
neurolab
math
matplotlib
time
````

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
import multilayernetworkbackprop

#define your training patterns here
zero = {'p':np.matrix([-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1]),'t': np.matrix([[1],
                                                                                                                        [0],
                                                                                                                        [0]])}
one = {'p': np.matrix([-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]), 't': np.matrix([[0],
                                                                                                                              [1],
                                                                                                                              [0]])}
two = {'p':np.matrix([1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1]), 't':np.matrix([[0],

                                                                                                                        [0],
trainingSet = [zero, one, two]                                                                                                                 [1]])}
#initalize the network
network = BackProp(epoch = 400, learning_rate=.10) # these are default values
network.train(trainingSet)

network.plotPerformance()
plt.show()

```
The training set can be any size any pattern, just has to be the same format list of dict,
where the keys' are 'p' and 't', respectivly

End with an example of getting some data out of the system or using it for a little demo



## Authors

* **Danny Ly** - *Initial work* - [RedKlouds](https://github.com/RedKlouds)


## Acknowledgments
* Learning
* Big Data analysis
* Mom?
* Inspiration
* fun


