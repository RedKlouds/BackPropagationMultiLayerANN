# Multi-Layer Perceptron  

This project features a 4 output neural network, using 2 layers the first layer uses a LogSigmoid activation function, the second layer uses a pure linear activation function.
---


-> finding as we can see the learning rate of a neural network using backpropogation is much more different than that of a Hebbian learning,
as a recall Hebbian learning algoirthm saves each pattern within its weights; think memorization. In a multi-layer neural network
we are training the network by showing it input, looking at the result, then propogating the amount of error back through the network to 
'correct' its behavior, an epoach is a training about through the entire training set going through the network and propogating back.

---
Training set for this project.
![trainingset](https://image.ibb.co/ju1jKb/Sample_IMages.png)


below are some resulting benchmarks to understand the networks behavior graphically\n
alpha .005 with 3 input patterns, over 400 epoachs
![.005](https://image.ibb.co/jrYSJR/005_alpha_3_patterns_perf.png)
![.005](https://image.ibb.co/hMHnJR/005_alpha_10_patterns_perf.png)
The top image shows the mean squared error during training the network to recognize
TOP: 3 digits
BOTTOM: 10 digits
we can see that the mean squared error during training even with 400 epoachs are nowhere near 50%
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
Below we have images the learning rate moved up to .1, over 400 epoachs with the same training samples as above
![.1](https://image.ibb.co/caWQ56/1_alpha_3_patterns_perf.png)
![.1](https://image.ibb.co/e4SbCm/1_alpha_10_patterns_perf.png)
TOP: training performance on .1 learning rate 3 Patterns
BOTTOM training performance on .1 learning rate 10 Patterns

As we can see and expect a much smoother curve of the mean squared error
coming down, to explain that the learning rate passed 20% error over the training set took about 100 epoachs only
which tells us we have convergence of error
->Results are expected since the learning rate in a backpropogated alogrithm tells the network how large of a step to take in the 
gradient decent inorder to minimize this error/loss function, the larger the step in this case of the data sample the quicker can achive a minimized
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


