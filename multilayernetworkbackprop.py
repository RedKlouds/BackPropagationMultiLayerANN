#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   multilayernetworkbackprop.py
#=#| Date:   11/16/2017
#=#|
#=#| Program Desc: This program is a Multi-Layered Artifical Neural Network, used to recognize
#=#| specific patterns, through pattern recognition, for this example the mainTest() function
#=#| uses 3 letters in the form of a row vector, which pertain to the letters '0','1', and '2'
#=#| in a 6X5 matrix, K, where i is the row, and j is the column, the training vector
#=#| index i,j is 1 when the current K(i,j) pixel is filled and -1 otherwise.
#=#| -> At the end of program execution call PlotPerformance to see the performance metrics
#=#| which summarize the number of epoachs vs learning mean squared error
#=#| -> Findings:
#=#|        * with a smaller learning rate, our gradient decent will take smaller steps in
#=#|        the direction of the gradient, however, smaller steps means it will learn MUCH
#=#|        slower, with a larger learning rate, learning is faster because the network can
#=#|        take much larger steps towards the gradient in each layer.
#=#|
#=#| Usage: Given a Epoach Traning iterations, and optional learning_rate make the Network
#=#| Object, then use train(p), a list of dicts which each dict is a training set, ie
#=#| p = [ {'p':row vector training set 1,'t':target column vector 1},
#=#|       {'p':row vector training set 2,'t':target column vector 2} ]
#=#| -> Call Perdict(p) where p, is the same format as above, to return a boolean pertaining
#=#|    to if the network was able to successfully classify the unseen data set, p.
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|


import numpy as np
from neurolab.trans import PureLin, LogSig, HardLim
import random
from matplotlib import pyplot as plt
########
# Note this program takes input vector in the form of the numpy matrix class
# Final Iteration
#####


class BackProp:
    def __init__(self, epoach, verbose=False, learning_rate =.10):
        """
        Lets first hard code a 2layer neural networks
        -Features 2 layers, with first layer having 2 neurons running logSIg and a final layer runnig purelin

        this is a Neural network that features 2 layers each layer, uses the purelin transferfunction
        The method we are trying to predict is, the default learning rate is .10
        """
        self.layers = list() # each layer will have a list, of stuff (container)
        self.error_result = None
        self.cur_error = 0
        self.epoach = epoach
        self.verbose = verbose
        self.learning_rate = learning_rate

    def setup2Layer(self,p, t):
        """
        *Perceptron Multi-Layered Network
        Set up the Neural Network default parameters, which include Number of Neurons per layer, and number of layer.
        For now a 2-layer Neural Network is initalized with random weights and biases
        :param p: Helps us get the number of weights nesscary for the network
        :param t: Helps us get the number of Neurons nessecary for the output layer
        :return: None
        """
        #TODO: This function will be made to adjust the number of neurons per layer, and number of layers, as well as the parameters for each layer such as weights and biass
        #work with everyhting as matrix not as arrays
        print("Setting unetwork the paramertsr are")
        p_row, p_col = p.shape
        num_nurons = 4
        num_inputs = 30
        print("P COLUM %s" % p_col)

        #layer_1_weights
        row, col = t.shape #given a matrix
        layer_1_weights = np.random.rand(num_nurons,num_inputs)
        layer_1_bias = np.random.rand(num_nurons,1)
        print("number of row for t%s"% row)

        #second layer is tricky the input(R) SXR , where R is the number of inputs from the first layer whcih is to asy
        #the same as the number of nurons each neuron gives an inpt
        #so the final layter the output we expect is a 3 nuron thingy [][][], where T is the target vector a column vector

        layer_2_weights = np.random.rand(row,num_nurons)
        layer_2_bias = np.random.rand(row,1)
        #to define the number of nerons S we will make a weight matrix that consist of SXR , where S is the number of neuron
        #and R is the number of inputs #30x1
        #define layer1. 4X30

        #change parameters of the transfer functiopn here
        self.layers.append({'weight':layer_1_weights, 'bias':layer_1_bias,'trans_func':'logsig'})# for layer 1
        self.layers.append( {'weight':layer_2_weights, 'bias':layer_2_bias,'trans_func':'logsig'}) # for layer 2


    def __dxSigMoid(self, a):
        """
        Logsigmoid transfer function derivative.
        :param a:
        :return:
        """
        return a * (1 - a)
    def __dxPureLin(self,a):
        return 1
    def __getDerivative(self, func_type):
        """
        return the derivative function
        :param func_type: a function transfer function, logsig, hardlim, purelin
        :return: derivative function of parameter func_type
        """
        lib = {'LOGSIG':self.__dxSigMoid, 'HARDLIM': HardLim(), 'PURELIN': self.__dxPureLin}
        return lib[func_type.upper()]
    def __getTransFunc(self,trans_type):
        """Initalizes and returns a transfer function object to use
        :param trans_type: 'logsig' , 'hardlim','purelim'
        :return: a transfer function object to call
        """
        #could make this its own item in the object self.trans['transType']
        lib = {'LOGSIG':LogSig(), 'HARDLIM': HardLim(), 'PURELIN':PureLin()}
        return lib[trans_type.upper()]
    def train(self, training_set=None):
        """
        takes a dict, where key is the input, and value is the target
        Main running function, 'fit' fit the current data model, or find a approximate function that fits this
        data , and model it.
        :return:
        """
        #epoach
        #get the first element of the trainig data,so we can set up the network with the dimension sizes
        self.setup2Layer(training_set[0]['p'], training_set[0]['t'])# get the first input vector to set up the weight and bias
        self.error_result = [0] * self.epoach

        for iterCount in range(self.epoach):
            for trainData in range(len(training_set)):
                self.forwardProp(training_set[trainData]['p'],training_set[trainData]['t'])
                self.error_result[iterCount] += (np.square(self.cur_error))
            #at the end of the epoach, we run through all patterns, and thbe error has been recorded, summed of sqaured errors in the current epoach index
            #lets make it the mean squared error
            self.error_result[iterCount] = self.error_result[iterCount]/len(training_set) # to get the mean of all errors summed
            if self.verbose:
                print(self.error_result)

    def forwardProp(self, p,t):
        """
        Remember we must run from the front to the back FIRST then go backwards
        this function will also add to the layers dictionary the respective output 'a' for each layer
        so we can refer to them back when we back propogate
        :param p: is a numpy array scaler or vector object
        :return:
        """
        #solve a for each forward layer and feed it to the next layer
        _p = p

        ####### static way
        #compute for layer 1

        ##################
        layerNum = 0
        for layer in self.layers:
            #get the transfer functions
            #for each layer in the network start at first
            tran_func = self.__getTransFunc(layer['trans_func'])#get the respective transfer function
            _weight = layer['weight']
            _bias = layer['bias']
            if layerNum == 0:
                #first layer
                _netinput = np.dot(_weight,_p.T) + _bias
            else:
                _netinput = np.dot(_weight,_p) + _bias
            layerNum +=1
            a = tran_func(_netinput) # this is the output for each layer we need to carriy i to the next
            #update,add the output value to the layer dict
            layer['a_output'] = a
            #layer


            _p = a # carry the output of this layer to the next layer
        self.output_a = _p

        #finished going forward save the error
        #e = t - a^M

        #error ^2 TODO

        #one iteration for 3 pattern update sum of error MSE
        #get the error square the error, store error, add each trainig sesssion running total / divide by the training session, MSE, STORE this into a global matrix or return and keep adding to this matrix

        # square the error add to the trainig session, at the end of the session divide by the number of training sesson to get the mean
        #this is one training plot, we need to add each time this function is run, which in our case is the number of input patters

        #add a running total of erros each time we train for each pattenrn in traiing set
        #then divide that by tnumber of pattern

        err = t - _p
        self.error = err #this is the global ERROR STEP 4 after runing the input through the network
        #we will then calculate the error to backpropogate into the network to produce the sensitivities


        self.cur_error = np.dot(err.T,err)
        #self.error_result.append(err)

        #step 2
        self.backProp(p)

        if self.verbose:
            print("=======================")
            #print("Final output of this 2-layer network is : %.3f | t is: %.3f\n\t\t**************ERROR is: %s" %(_p, t, self.error))
            print("========================")

    def predict(self, p, t, ret_actual_a = False):
        """

        :param p: is a row vector matrix, of a pattern
        :param t: is a column matrix of the target of the pattern given
        :param ret_actual_a: if true will return the actual output of a, computed by
        the final layer.
        :return: Depending on the flat ret_actual_a, a boolean for if the netowrk correctly classified
        the sample, or the actual returned value of a
        """
        # solve a for each forward layer and feed it to the next layer
        _p = p['p']

        layerNum = 0
        for layer in self.layers:
            # get the transfer functions
            # for each layer in the network start at first
            tran_func = self.__getTransFunc(layer['trans_func'])  # get the respective transfer function
            _weight = layer['weight']
            _bias = layer['bias']
            if layerNum == 0:
                # first layer
                _netinput = np.dot(_weight, _p.T) + _bias
            else:
                _netinput = np.dot(_weight, _p) + _bias
            layerNum += 1
            a = tran_func(_netinput)  # this is the output for each layer we need to carriy i to the next
            # update,add the output value to the layer dict
            _p = a  # carry the output of this layer to the next layer

        err = t - _p
        self.error = err  # this is the global ERROR STEP 4 after runing the input through the network
        # we will then calculate the error to backpropogate into the network to produce the sensitivities
        self.cur_error = np.dot(err.T, err)

        purelim = PureLin()
        hardlim = HardLim()

        result = hardlim(_p -.5) # returns boolean same or not

        print("Result: %s Target %s"%(result,t) )
        if not ret_actual_a:
            return (result == t).all() #boolean true or not
        return result
    def __getDxMatrix(self,index):
        """
        Returns a matrix corresonpding to the jacobian matrix for derivative matrix
        F^1(n^1)
        Thjis function also know the transfer function from the index parameter
        which gives it the transfer function type, which we can compute the matrix, and each dict R
        REQIRES that forward feed is run PRIOR TO RUNNING THIS FUCNTION
        so that every layer hs a respective output value to access else keyerror, TODO catch
        KOWING that the output a is alwasyt a SX1 , where S is the number of nurons , and its always a colum
        vector we can safly access it like we do diagnoally
        :param index:
        :return:
        """
        num_neurons = len(self.layers[index]['a_output']) # this will give me the SXR matrix , where S tells me the
        #number of nurons, which also tell me how to make the jacobian Matrix for the derivative matrix
        jaccob_matrix = np.zeros(shape=(num_neurons,num_neurons)) # ie S=3, shape 3X3

        #get the transfer function type and compute using the a value stored

        dx_func = self.__getDerivative(self.layers[index]['trans_func']) #jsut call like dx_func(value)
        for i in range(num_neurons):
            #diagonal matrix
            a_val = self.layers[index]['a_output'][i] # get the respective ith element in the a vecotr
            jaccob_matrix[i][i] = dx_func(a_val)#derivative transfer function compute
        return jaccob_matrix

    def backProp(self,p):
        s_final = -2
        #go backwards remember DIS
        #goes from the length which is NOT index based
        #for i in reversed(self.layers):
        _saveSensitivitiy = 0
        for i in range(len(self.layers)-1, -1, -1):
            #Jacobian Matrix, here
            dx_f_matrix = self.__getDxMatrix(index=i)  # assumes we get this value
            if i == (   len(self.layers) - 1 ): # Check if last layer
                #current last Layer Generate inital Sensitivity
                #dx_f_matrix = self.__getDxMatrix(index=i)#assumes we get this value
                #muliply the weight and sensitivity first

                #S^M = -2F*^M(n^M)(t-a) [Error]

                _saveSensitivitiy = (-2) * dx_f_matrix * self.error #current error from the current iteration, to backprop
                #_saveSensitivitiy = _sensitivity
            else:

                _w  = self.layers[i+1]['weight']
                sens = np.dot(_w.T , _saveSensitivitiy)
                _saveSensitivitiy = np.dot(dx_f_matrix, sens)

                #update and save the sensitivity for the current layer for updating weights and biases
            self.layers[i]['sensitivity'] = _saveSensitivitiy
            if self.verbose:
                print("THIS IS S" , _saveSensitivitiy)
            #step 3
        self.trainWeights(p)#retrain the weights and bias
    def trainWeights(self,p):
        """
        Retrain the weights CALLED AS STEP 3 AFTER ForwardProp, BackProp
        :param learning_rate_alpha: This is a floating point number for the leraning rate at which we shoul update
        the weights, we want a small learning rate
        :return:
        """
        #train the weights from last layer to first layer
        output_a = 0
        for layer in range(len(self.layers)-1,-1,-1):
            lay_w_old = self.layers[layer]['weight']#get the weight
            lay_sensitivity = self.layers[layer]['sensitivity']
            if layer == 0:
                #first inital layer a must be the input vector
                #can change to have 3 layers the zero'th layer is the inital layer with the inital p and t values
                #therefore making a loop iterate backwards but skipping the first layer, but that for later TODO
                #technically 3 layers
                output_a = p
            else:
                output_a = self.layers[layer-1]['a_output'].T
            self.layers[layer]['weight'] = ( (lay_w_old) - ( (self.learning_rate) * lay_sensitivity * output_a))
            #self.layers[layer]['weight'] = self.layers[layer]['weight'] - ( (learning_rate_alpha) * self.layers[layer]['sensitivity'] * self.layers[layer-1]['a_output'].T)
            #update the bias
            self.layers[layer]['bias'] = self.layers[layer]['bias'] - ( (self.learning_rate) * self.layers[layer]['sensitivity'])
            if self.verbose:
                print("bias for layer %s updated %s"% (layer, self.layers[layer]['bias']))

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
        Given the error_result list, where each index is a epoach or a time we trained,
        plot the errors on a grapp
        :return:
        """

        plt.figure(1)
        for i in range(len(self.error_result)):

            self.error_result[i] = self.error_result[i][0,0]
            #print(self.error_result[i])

        plt.plot(self.error_result)
        plt.title(r"Mean Square Error results, $\alpha  %.4f" % self.learning_rate)# % self.learning_rate)
        plt.xlabel("Epochs, iterations")
        plt.ylabel("mean Squared error")
        plt.ylim([0,1])
        plt.xlim([0,self.epoach])
        #plt.show()


def generateNoise(original, pixelToChange):
    copyMax = original['p'].copy()#make a hard copy
    randomNums = np.random.permutation(pixelToChange)

    for index in randomNums:
        copyMax[0,index] = (copyMax[0,index] * -1)

    return {'p':copyMax}


def test():


    zero = {'p':np.matrix([-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1]),'t': np.matrix([[1],
                                                                                                                        [0],
                                                                                                                        [0]])}
    one = {'p': np.matrix([-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]), 't': np.matrix([[0],
                                                                                                                              [1],
                                                                                                                              [0]])}
    two = {'p':np.matrix([1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,-1,1]), 't':np.matrix([[0],
                                                                                                                        [0],
                                                                                                                        [1]])}
    FinalTestData = [zero,one,two]


    brokenZero = generateNoise(zero, 8) # returns a dict {'p': np.matrix([])}

    network = BackProp(epoach=700, learning_rate = .1)

    network.train(FinalTestData)

    print("Predicting...: are they the same ? %s" % network.predict(brokenZero, zero['t']) )
    #plot the networks Performance
    network.plotPerformance()


    #create the performance over the interval of 4 test,
    #x axis will be [0,1,2,3,4]
    #where 0:0, 1:2, 2:4, 3:6, 4:8 pixels removed
    #so y_list will be a size of 5 each index has the respective items
    y_plot = [0] * 5
    pixel_remove= [0,2,4,6,8]
    #same size as y_lot and x_axis
    bar_x = np.arange(0,len(pixel_remove))
    x_axis_bar = [0, 2, 4, 6, 8]

    numTest = 50
    for iii in range(5):
        #for each x axis
        for j in range(numTest):
            #run the test 50 times
            #grab the random input vector to change
            ranindex = random.randint(0, len(FinalTestData)-1)
            change = generateNoise(FinalTestData[ranindex], x_axis_bar[iii])  # get rangom vector and chagne pix times
            performance = network.predict(change, FinalTestData[ranindex]['t'])
            if performance:
                y_plot[iii] += 1
        y_plot[iii] = 1 - (y_plot[iii] / numTest)



    figz = plt.figure(2)
    #ax = figz.add_subplot(1,1,1)
    #ax.set_xticks(x_axis)
    plt.title("Performance with respective Noise")
    plt.bar(x_axis_bar,y_plot)
    plt.xlabel("Number of Pixels changed")
    plt.ylabel("Mean performance, per %s iterations per pixel" % numTest)
    plt.grid()
    plt.show()

    plt.close()


if __name__ == "__main__":
    test()