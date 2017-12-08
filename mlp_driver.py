# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
# =#| Author: Danny Ly MugenKlaus|RedKlouds
# =#| File:   mlp_driver.py
# =#| Date:   12/7/2017
# =#|
# =#| Program Desc:
# =#|
# =#| Usage:
# =#|
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

import random
import numpy as np
from matplotlib import pyplot as plt
from multilayerperceptron import MultiLayerPerceptron


def generateNoise(original, pixelToChange):
    """
    Transform the original image vectors with noise, given the number
    of vectors to change.
    :param original: Original Vector
    :param pixelToChange: Number of pixels or indexs to change
    :return: Changed vector
    """
    copyMax = original['p'].copy()  # make a hard copy
    randomNums = np.random.permutation(pixelToChange)

    for index in randomNums:
        copyMax[0, index] = (copyMax[0, index] * -1)

    return {'p': copyMax}


def testDataPerformance(network, data, learningRate, numSamples, epochs):
    """
    Takes the data parameter and introduces noise, via changes the pixels randomly within the image
    :param network:
    :param data:
    :return:
    """
    # create the performance over the interval of 4 test,
    # x axis will be [0,1,2,3,4]
    # where 0:0, 1:2, 2:4, 3:6, 4:8 pixels removed
    # so y_list will be a size of 5 each index has the respective items
    y_plot = [0] * 5
    x_axis_bar = [0, 2, 4, 6, 8]

    numTest = 400
    for iii in range(5):  # for each pixel to remove category
        # for each x axis
        for j in range(numTest):  # for each test
            # run the test 50 times
            # grab the random input vector to change
            ranindex = random.randint(0, len(data) - 1)
            # grab a random input to apply noise to
            change = generateNoise(data[ranindex], x_axis_bar[iii])  # get random vector and change pix times
            performance = network.predict(change, data[ranindex]['t'])
            if performance:
                y_plot[iii] += 1
        y_plot[iii] = 1 - (y_plot[iii] / numTest)

    figz = plt.figure(2)

    plt.title(r"Test set Performance $\alpha$=%.3f $\eta$=%s epochs=%s" % (learningRate, numSamples, epochs))
    plt.bar(x_axis_bar, y_plot)
    plt.xlabel("Number of Pixels changed")
    plt.ylabel("Mean performance per %s iterations per pixel" % numTest)
    plt.grid()
    plt.show()


def getData():
    zero = {'p': np.matrix([-1, 1, 1, 1, 1, -1,
                            1, -1, -1, -1, -1, 1,
                            1, -1, -1, -1, -1, 1,
                            1, -1, -1, -1, -1, 1,
                            -1, 1, 1, 1, 1, -1]),
            't': np.matrix([[0],
                            [0],
                            [0],
                            [0]])}
    one = {'p': np.matrix([-1, -1, -1, -1, -1, -1,
                           1, -1, -1, -1, -1, -1,
                           1, 1, 1, 1, 1, 1,
                           -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1]),
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

    three = {'p': np.matrix([1, -1, -1, -1, -1, 1,
                             1, -1, 1, 1, -1, 1,
                             1, -1, 1, 1, -1, 1,
                             1, 1, 1, 1, 1, 1,
                             -1, -1, -1, -1, -1, -1]),
             't': np.matrix([[0],
                             [0],
                             [1],
                             [1]])}
    four = {'p': np.matrix([-1, -1, -1, 1, -1, -1,
                            -1, -1, 1, 1, -1, -1,
                            -1, 1, -1, 1, -1, -1,
                            1, 1, 1, 1, 1, 1,
                            -1, -1, -1, 1, -1, -1]),
            't': np.matrix([[0],
                            [1],
                            [0],
                            [0]])}
    five = {'p': np.matrix([1, 1, 1, -1, -1, 1,
                            1, -1, 1, -1, -1, 1,
                            1, -1, 1, -1, -1, 1,
                            1, -1, 1, 1, 1, 1,
                            -1, -1, -1, -1, -1, -1]),
            't': np.matrix([[0],
                            [1],
                            [0],
                            [1]])}
    six = {'p': np.matrix([1, 1, 1, 1, 1, 1,
                           1, -1, -1, 1, -1, 1,
                           1, -1, -1, 1, -1, 1,
                           1, -1, -1, 1, 1, 1,
                           -1, -1, -1, -1, -1, -1]),
           't': np.matrix([[0],
                           [1],
                           [1],
                           [0]])}
    seven = {'p': np.matrix([1, -1, -1, -1, -1, 1,
                             1, -1, -1, -1, 1, -1,
                             1, -1, -1, 1, -1, -1,
                             1, -1, 1, -1, -1, -1,
                             1, 1, -1, -1, -1, -1]),
             't': np.matrix([[0],
                             [1],
                             [1],
                             [1]])}
    eight = {'p': np.matrix([1, 1, 1, 1, 1, 1,
                             1, -1, 1, -1, -1, 1,
                             1, -1, 1, -1, -1, 1,
                             1, -1, 1, -1, -1, 1,
                             1, 1, 1, 1, 1, 1]),
             't': np.matrix([[1],
                             [0],
                             [0],
                             [0]])}
    nine = {'p': np.matrix([1, 1, 1, -1, -1, -1,
                            1, -1, 1, -1, -1, -1,
                            1, -1, 1, -1, -1, -1,
                            1, 1, 1, 1, 1, 1,
                            -1, -1, -1, -1, -1, -1]),
            't': np.matrix([[1],
                            [0],
                            [0],
                            [1]])}
    FinalTestData = [zero, one, two, three, four, five, six, seven, eight, nine]
    return FinalTestData


def test():
    FinalTestData = getData()
    test_epoch = 100
    test_learning_rate = .5
    test_number_sample = len(FinalTestData)

    network = MultiLayerPerceptron(epoch=test_epoch, learning_rate=test_learning_rate)

    network.train(FinalTestData)

    network.plotPerformance()

    testDataPerformance(network, FinalTestData, test_learning_rate, test_number_sample, test_epoch)

    plt.close()


if __name__ == "__main__":
    test()
