# Multi-Layer Neural Network with Backpropagation  

This project features a 4 output neural network, using 2 layers the first layer uses a LogSigmoid activation function, the second layer uses a pure linear activation function.
---


This project is a implementation of the multi layer neural network, instead of the Hebbian learning rule implemented
before this project, was made to solve the same problem , which is pattern recognition. Instead of a single layered
preceptron learning rule where the patterns are memorized in the weights. the mutli-layer network will use <i>Back propogation</i>
to estimate the function g(x) or 'problem'. In a multi-layer one method used to train the network would be to use a
Back propogation, in other words once the network has seen the training data, we record the summed mean squared errors(summer)
and propagate (feed-back) into the network those errors, such that the network will recognize its errors and readjust itself.
to not get so math-friendly, we know that the transfer/activation function is used to produce the linear relationship, (thus we need
to keep in mind choosing the hidden layers, must contain linear transfer functions), because the feed-forward layers used a linear
transfer function to build a relationship between the input vectors points. Because our network realtionship functions are linear
we can also derive that, to get a point in which a linear function (f) tells use that derivative of f -> f^ of input x will be the instantaneous
change in the function f^ where tells our network how close we are from the minima, meaning when f^ ==0 we are at a zero slope
with no changes, instead of estimating the value at x we want to estimate the change in the error function at x, this is because
we want the network to


-> finding as we can see the learning rate of a neural network using backpropogation is much more different than that of a Hebbian learning,
as a recall Hebbian learning algoirthm saves each pattern within its weights; think memorization. In a multi-layer neural network
we are training the network by showing it input, looking at the result, then propogating the amount of error back through the network to 
'correct' its behavior, an epoach is a training about through the entire training set going through the network and propogating back.



below are some resulting benchmarks to understand the networks behavior graphically\n
alpha .005 with 3 input patterns, over 400 epoachs\n
![.005](https://image.ibb.co/jrYSJR/005_alpha_3_patterns_perf.png)
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


