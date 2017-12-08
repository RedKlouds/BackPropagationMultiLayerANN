#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   mlp_driver.py
#=#| Date:   12/7/2017
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

from multilayerperceptron import MultiLayerPerceptron

from matplotlib import pyplot as plt

import numpy as np

import random

def generateNoise(original, pixelToChange):
    """
    Transform the original image vectors with noise, given the number
    of vectors to change.
    :param original: Original Vector
    :param pixelToChange: Number of pixels or indexs to change
    :return: Changed vector
    """
    copyMax = original['p'].copy()#make a hard copy
    randomNums = np.random.permutation(pixelToChange)

    for index in randomNums:
        copyMax[0,index] = (copyMax[0,index] * -1)

    return {'p':copyMax}

def test():


    zero = {'p':np.matrix([-1,1,1,1,1,-1,
                           1,-1,-1,-1,-1,1,
                           1,-1,-1,-1,-1,1,
                           1,-1,-1,-1,-1,1,
                           -1,1,1,1,1,-1]),
            't':np.matrix([[0],
                            [0],
                            [0],
                            [0]])}
    one = {'p': np.matrix([-1,-1,-1,-1,-1,-1,
                           1,-1,-1,-1,-1,-1,
                           1,1,1,1,1,1,
                           -1,-1,-1,-1,-1,-1,
                           -1,-1,-1,-1,-1,-1]),
           't': np.matrix([[0],
                           [0],
                           [0],
                           [1]])}

    two = {'p': np.matrix([1, -1, -1, -1, -1, -1,
                           1, -1, -1, 1, 1, 1,
                           1, -1, -1, 1, -1, 1,
                           -1, 1, 1, -1, -1, 1,
                           -1, -1, -1, -1, -1, 1]),
           't': np.matrix([[0],
                           [0],
                           [1],
                           [0]])}

    three = {'p':np.matrix([1,-1,-1,-1,-1,1,
                            1,-1,1,1,-1,1,
                            1,-1,1,1,-1,1,
                            1,1,1,1,1,1,
                            -1,-1,-1,-1,-1,-1]),
             't':np.matrix([[0],
                            [0],
                            [1],
                            [1]])}
    four = {'p':np.matrix([-1,-1,-1,1,-1,-1,
                           -1,-1,1,1,-1,-1,
                           -1,1,-1,1,-1,-1,
                           1,1,1,1,1,1,
                           -1,-1,-1,1,-1,-1]),
            't':np.matrix([[0],
                           [1],
                           [0],
                           [0]])}
    five = {'p':np.matrix([1,1,1,-1,-1,1,
                           1,-1,1,-1,-1,1,
                           1,-1,1,-1,-1,1,
                           1,-1,1,1,1,1,
                           -1,-1,-1,-1,-1,-1]),
            't':np.matrix([[0],
                           [1],
                           [0],
                           [1]])}
    six = {'p':np.matrix([1,1,1,1,1,1,
                          1,-1,-1,1,-1,1,
                          1,-1,-1,1,-1,1,
                          1,-1,-1,1,1,1,
                          -1,-1,-1,-1,-1,-1]),
           't':np.matrix([[0],
                         [1],
                         [1],
                         [0]])}
    seven={'p':np.matrix([1,-1,-1,-1,-1,1,
                          1,-1,-1,-1,1,-1,
                          1,-1,-1,1,-1,-1,
                          1,-1,1,-1,-1,-1,
                          1,1,-1,-1,-1,-1]),
           't':np.matrix([[0],
                          [1],
                          [1],
                          [1]])}
    eight={'p':np.matrix([1, 1, 1, 1, 1, 1,
                          1, -1, 1, -1, -1, 1,
                          1, -1, 1, -1, -1, 1,
                          1, -1, 1, -1, -1, 1,
                          1, 1, 1, 1, 1, 1]),
           't':np.matrix([[1],
                         [0],
                         [0],
                         [0]])}
    nine={'p':np.matrix([1,1,1,-1,-1,-1,
                         1,-1,1,-1,-1,-1,
                         1,-1,1,-1,-1,-1,
                         1, 1, 1, 1, 1 ,1,
                         -1,-1,-1,-1,-1,-1]),
          't':np.matrix([[1],
                         [0],
                         [0],
                         [1]])}
    FinalTestData = [zero,one,two, three, four, five, six, seven, eight, nine]


    brokenZero = generateNoise(zero, 8) # returns a dict {'p': np.matrix([])}

    network = MultiLayerPerceptron(epoch=400, learning_rate = 2)

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

    numTest = 400
    for iii in range(5): # for each pixel to remove category
        #for each x axis
        for j in range(numTest):#for each test
            #run the test 50 times
            #grab the random input vector to change
            ranindex = random.randint(0, len(FinalTestData)-1)
            #grab a random input to apply noise to
            change = generateNoise(FinalTestData[ranindex],x_axis_bar[iii])  # get random vector and change pix times
            performance = network.predict(change, FinalTestData[ranindex]['t'])
            if performance:
                y_plot[iii] += 1
        y_plot[iii] = 1 - (y_plot[iii] / numTest)



    figz = plt.figure(2)
    #ax = figz.add_subplot(1,1,1)
    #ax.set_xticks(x_axis)
    plt.title("Neural Network performance with Noise")
    plt.bar(x_axis_bar,y_plot)
    plt.xlabel("Number of Pixels changed")
    plt.ylabel("Mean performance per %s iterations per pixel" % numTest)
    plt.grid()
    plt.show()

    plt.close()

if __name__ == "__main__":
    test()