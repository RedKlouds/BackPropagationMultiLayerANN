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

### Architecture 
![Architecture](https://image.ibb.co/nr5HFG/Artchetectre.png)\

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

### Results

During the discovery of this network's behavior one very important hyper parameter that must be taking into account is the
learning rate. As the back-propagation works by taking the derivative of the activation function to find the steepest decent
of the loss function (activation function). The final step is to relay that error into the networks weights to readjust the neurons
to the correct locations in the input space.

This is done through a multiplication of the neurons current location by the error amount
multiplied by the learning rate( scalar value ) when the scalar value is large the network will 'step over' the optimized minimum loss function
causing worst performance, however if the learning rate is too small, the steps in the steepest gradient will take very long to converge, therefore
training will see incomplete since the network was not able to converge to their respective hyperplanes.

Low Learning Rate       |   Higher Learning Rate
:----------------------:|:------------------------:
![.005](https://image.ibb.co/b57LkG/A_1_N10_E100_Error.png) |![.005](https://image.ibb.co/fERfkG/A_5_N10_E100_Error.png)
The above figure suggest using a leraning rate that is too low will not allow the network to properly completely learn.
Looking at the figure we also notice that with a smaller learning rate however its clear the learning steps are much smoother
i.e. the smoother curve verse the lower mse on the right however the learning curve is much more rigid.

This problem introduces more methods when it comes to neural networks. Generalization, generalizations are methods employed to 
neural networks to help find the optimum learning rate. It is necessary since a small learning rate will allow us to converge the problem
with convergence is, if we converge to the training set, our network will be over-fitted to our training data.

In the project I was able to implement one method of the generalization techniques called 'early stopping', here we have our data set
split into 3 sets, a test set, validation set, and training set. we train the network and watch as the networks mse on the training set converge
as normal, however as we are testing the mse we also run our validation data set on the network, as the network converges there lies a point at which
the network performance on the validation set will start to 're bound' which will signify the point the network does not recognize the data in the validation set
this is where we 'early stop' our training. Doing so will allow our network to be more general to the points around it, making it approximate a much 
similar function f(x)

- unfortunately i do not supply any plots for the early stopping, and will continue to work to provides them  as my research progresses

Lower learning rate | Higher learning rate
:------------------:|:--------------------------:
[](https://image.ibb.co/dFWPCw/A_1_N10_E100_Test.png)|[](https://image.ibb.co/cQSmQG/A_5_N10_E100_Test.png)
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


