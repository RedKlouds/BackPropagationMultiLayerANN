# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
# =#| Author: Danny Ly MugenKlaus|RedKlouds
# =#| File:   multilayerperceptron.py
# =#| Date:   11/16/2017
# =#|
# =#| Program Desc: This program is a Multi-Layered Artifical Neural Network, used to recognize
# =#| specific patterns, through pattern recognition, for this example the mainTest() function
# =#| uses 3 letters in the form of a row vector, which pertain to the letters '0','1', and '2'
# =#| in a 6X5 matrix, K, where i is the row, and j is the column, the training vector
# =#| index i,j is 1 when the current K(i,j) pixel is filled and -1 otherwise.
# =#| -> At the end of program execution call PlotPerformance to see the performance metrics
# =#| which summarize the number of epoachs vs learning mean squared error
# =#| -> Findings:
# =#|        * with a smaller learning rate, our gradient decent will take smaller steps in
# =#|        the direction of the gradient, however, smaller steps means it will learn MUCH
# =#|        slower, with a larger learning rate, learning is faster because the network can
# =#|        take much larger steps towards the gradient in each layer.
# =#|
# =#| Usage: Given a epoch Traning iterations, and optional learning_rate make the Network
# =#| Object, then use train(p), a list of dicts which each dict is a training set, ie
# =#| p = [ {'p':row vector training set 1,'t':target column vector 1},
# =#|       {'p':row vector training set 2,'t':target column vector 2} ]
# =#| -> Call Perdict(p) where p, is the same format as above, to return a boolean pertaining
# =#|    to if the network was able to successfully classify the unseen data set, p.
# =#|
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|


import numpy as np
from matplotlib import pyplot as plt

from neurotrans import LogSig, HardLim, PureLin

class MultiLayerPerceptron:
    def __init__(self, epoch, verbose=False, learning_rate=.10):
        """
        Lets first hard code a 2layer neural networks
        -Features 2 layers, with first layer having 2 neurons running logSIg and a final layer runnig purelin

        this is a Neural network that features 2 layers each layer, uses the purelin transferfunction
        The method we are trying to predict is, the default learning rate is .10
        """
        self.layers = list()  # each layer will have a list, of stuff (container)
        self.error_result = [0] * epoch  # initalize record keeping of error
        self.cur_error = 0
        self.num_epoch = epoch
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.numSamples = 0

        self.dannyErrors = list()

    def _initNetwork(self, p, t):
        """
        Initialize the Weight and biases of the network, with uniform random values.
        :param p: size of input vector
        :param t: size of target vector
        :return: None
        """
        # TODO: This function will be made to adjust the number of neurons per layer, and number of layers, as well as the parameters for each layer such as weights and biass
        # work with everyhting as matrix not as arrays
        print("Setting unetwork the paramertsr are")
        p_row, p_col = p.shape
        num_neurons = 50
        num_inputs = 30
        print("P COLUM %s" % p_col)

        # layer_1_weights
        row, col = t.shape  # given a matrix
        layer_1_weights = np.random.rand(num_neurons, num_inputs)
        layer_1_bias = np.random.rand(num_neurons, 1)
        print("number of row for t%s" % row)

        # second layer is tricky the input(R) SXR , where R is the number of inputs from the first layer whcih is to asy
        # the same as the number of nurons each neuron gives an inpt
        # so the final layter the output we expect is a 3 nuron thingy [][][], where T is the target vector a column vector

        layer_2_weights = np.random.rand(row, num_neurons)
        layer_2_bias = np.random.rand(row, 1)
        # to define the number of nerons S we will make a weight matrix that consist of SXR , where S is the number of neuron
        # and R is the number of inputs #30x1
        # define layer1. 4X30

        # change parameters of the transfer functiopn here
        self.layers.append({'weight': layer_1_weights, 'bias': layer_1_bias, 'trans_func': 'logsig'})  # for layer 1
        self.layers.append({'weight': layer_2_weights, 'bias': layer_2_bias, 'trans_func': 'logsig'})  # for layer 2

    def __dxSigMoid(self, a):
        """
        Logsigmoid transfer function derivative.
        :param a: output a
        :return: returns derivative of log Sigmoid
        """
        return a * (1 - a)

    def __dxPureLin(self, a):
        """
        :param a: output a 
        :return: returns derivative of linear function
        """
        return 1

    def __getDerivative(self, func_type):
        """
        return the derivative function
        :param func_type: a function transfer function, logsig, hardlim, purelin
        :return: derivative function of parameter func_type
        """
        #lib = {'LOGSIG': self.__dxSigMoid, 'HARDLIM': HardLim(), 'PURELIN': self.__dxPureLin}
        lib = {'LOGSIG': LogSig(), 'HARDLIM': HardLim(), 'PURELIN': PureLin().derivative}
        return lib[func_type.upper()]

    def __getTransFunc(self, trans_type):
        """
        Initializes and returns a transfer function object to use
        :param trans_type: 'logsig' , 'hardlim','purelim'
        :return: a transfer function object to call
        """
        # could make this its own item in the object self.trans['transType']
        lib = {'LOGSIG': LogSig(), 'HARDLIM': HardLim(), 'PURELIN': PureLin()}
        return lib[trans_type.upper()]

    def train(self, training_set=None):
        """
        takes a dict, where key is the input, and value is the target
        :return: None
        """
        # initalize weights and bias with input vector size, and target size
        self._initNetwork(training_set[0]['p'], training_set[0]['t'])

        self.numSamples = len(training_set)

        for epoch in range(self.num_epoch):
            self.epochError = list()
            for trainingData in range(len(training_set)):
                self._feedForward(training_set[trainingData]['p'], training_set[trainingData]['t'])

                self.error_result[epoch] += (np.square(self.cur_error))

            # record the mean squared error of the current epoch
            self.error_result[epoch] = self.error_result[epoch] / len(training_set)

            x = self.epochError[0]
            for i in range(1, len(self.epochError)):
                x += self.epochError[i]
            x = x.mean()
            self.dannyErrors.append(((1 / len(training_set)) * x))
            # self.dannyErrors.append( x -.5)

    def _feedForward(self, p, t):
        """
        feeds the input vector forward through the network saving the error.
        :param p: is a numpy array scalar or vector object
        :return: None
        """
        _p = p
        layerNum = 0
        for layer in self.layers:
            # current layers data
            tran_func = self.__getTransFunc(layer['trans_func'])
            _weight = layer['weight']
            _bias = layer['bias']
            if layerNum == 0:
                # first layer
                _netinput = np.dot(_weight, _p.T) + _bias
            else:
                _netinput = np.dot(_weight, _p) + _bias
            layerNum += 1
            a = tran_func(_netinput)

            layer['a_output'] = a
            # layer
            _p = a  # input for following layers are the output to the previous layers

        self.output_a = _p

        error = np.square(t - _p)
        self.epochError.append(error)

        err = t - p
        self.error = err

        err = t - _p
        self.error = err
        self.cur_error = np.dot(err.T, err)
        # step 2

        self._backPropagate(p)

    def predict(self, p, t, ret_actual_a=False):
        """
        :param p: is a row vector matrix, of a pattern
        :param t: is a column matrix of the target of the pattern given
        :param ret_actual_a: if true will return the actual output of a, computed by
        the final layer.
        :return: Depending on the flat ret_actual_a, a boolean for if the network correctly classified
        the sample, or the actual returned value of a
        """
        # solve a for each forward layer and feed it to the next layer
        _p = p['p']

        layerNum = 0
        for layer in self.layers:
            tran_func = self.__getTransFunc(layer['trans_func'])
            _weight = layer['weight']
            _bias = layer['bias']
            if layerNum == 0:
                # first layer
                _netinput = np.dot(_weight, _p.T) + _bias
            else:
                _netinput = np.dot(_weight, _p) + _bias
            layerNum += 1
            # this is the output for each layer we need to carry i to the next
            a = tran_func(_netinput)
            # update,add the output value to the layer dict
            _p = a  # carry the output of this layer to the next layer

        err = t - _p
        self.error = err  # this is the global ERROR STEP 4 after runing the input through the network
        self.cur_error = np.dot(err.T, err)

        hardlim = HardLim()

        result = hardlim(_p - .5)  # returns boolean same or not

        print("Result: %s Target %s" % (result, t))
        if not ret_actual_a:
            return (result == t).all()  # boolean true or not
        return result

    def __getDxMatrix(self, index):
        """
        Computes the corresponding jaccbian matrix for the derivative matrix
        F^1(n-1)

        Precondition: feed forward must've been run prior to this function call
        :param index:
        :return: None
        """
        num_neurons = len(self.layers[index]['a_output'])
        jaccob_matrix = np.zeros(shape=(num_neurons, num_neurons))  # ie S=3, shape 3X3
        dx_func = self.__getDerivative(self.layers[index]['trans_func'])
        for i in range(num_neurons):
            # diagonal matrix
            a_val = self.layers[index]['a_output'][i]
            jaccob_matrix[i][i] = dx_func(a_val)
        return jaccob_matrix

    def _backPropagate(self, p):
        """
        Propagates sensitivities of the error back through the network

        Precondition: feed forward must've been run prior to this function call
        :param p: original input vector
        :return: None
        """
        _saveSensitivitiy = 0
        for i in range(len(self.layers) - 1, -1, -1):
            # Jacobian Matrix, here
            dx_f_matrix = self.__getDxMatrix(index=i)
            if i == (len(self.layers) - 1):  # Check if last layer
                # initial sensitivity
                # S^M = -2F*^M(n^M)(t-a) [Error]
                _saveSensitivitiy = (-2) * dx_f_matrix * self.error
            else:
                _w = self.layers[i + 1]['weight']
                sens = np.dot(_w.T, _saveSensitivitiy)
                _saveSensitivitiy = np.dot(dx_f_matrix, sens)
                # update and save the sensitivity for the current layer for updating weights and biases
            self.layers[i]['sensitivity'] = _saveSensitivitiy
            if self.verbose:
                print("THIS IS S", _saveSensitivitiy)
                # step 3
        self._updateWeights(p)  # retrain the weights and bias

    def _updateWeights(self, p):
        """
        Precondition: forwardfeed and backpropogate has been called prior to this function call
        :param p: initial training vector
        :return: None
        """
        # train the weights from last layer to first layer
        output_a = 0
        for layer in range(len(self.layers) - 1, -1, -1):
            lay_w_old = self.layers[layer]['weight']  # get the weight
            lay_sensitivity = self.layers[layer]['sensitivity']
            if layer == 0:
                # first inital layer a must be the input vector
                # can change to have 3 layers the zero'th layer is the inital layer with the inital p and t values
                # therefore making a loop iterate backwards but skipping the first layer, but that for later TODO
                # technically 3 layers
                output_a = p
            else:
                output_a = self.layers[layer - 1]['a_output'].T
            self.layers[layer]['weight'] = ((lay_w_old) - ((self.learning_rate) * lay_sensitivity * output_a))

            self.layers[layer]['bias'] = self.layers[layer]['bias'] - (
                (self.learning_rate) * self.layers[layer]['sensitivity'])

    def reportStats(self):
        x = ''
        for layer in range(len(self.layers)):
            x += "\t\t=======Layer: %s\n" % layer
            x += "Weight: \n%s\n" % self.layers[layer]['weight']
            x += "bias: \n%s\n" % self.layers[layer]['bias']
            if 'a_output' in self.layers[layer]:
                x += "a: %s\n" % self.layers[layer]['a_output']
            if 'sensitivity' in self.layers[layer]:
                x += 'sensitivity: %s\n' % self.layers[layer]['sensitivity']
            x += '\n'
        return x

    def plotPerformance(self):
        """
        Plot the current mean squared error
        :return: None
        """

        fig = plt.figure(69)
        plt.plot(self.dannyErrors, label="MLP mse", c='#EC7063')
        plt.title(r"Mean Squared Error(mse) $\alpha$%.3f $\eta$=%s" % (self.learning_rate, self.numSamples))
        plt.xlabel("Epoch iterations")
        plt.ylabel("Error (percentage %)")
        plt.legend()
        plt.xlim([0, self.num_epoch])
        plt.ylim([0, 1])

        plt.show()


if __name__ == "__main__":
    print("Called main")
