# Multi-Layer Neural Network with Backpropagation  

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
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

