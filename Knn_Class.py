import collections
from statistics import mode
import numpy as np
import collections
from itertools import islice

def classify_nn(training_filename, testing_filename, k):
    # convert data into numpy vectors

    # convert training data
    training_stack = []
    training_goal = []
    testing_stack = []
    predicted_goal = []
    with open(training_filename) as training_data:
        for line in training_data.readlines():
            linerow = line.split(",")
            training_goal.append(linerow[-1])
            linerow.pop()
            listrow = np.array(linerow).astype(float)
            training_stack.append(listrow)
    training_matrix = np.vstack(training_stack)

    # convert testing data
    with open(testing_filename) as testing_data:
        for line in testing_data.readlines():
            linerow = line.split(",")
            #if "\n" in linerow[-1]:
            #    linerow.pop()
            listrow = np.array(linerow).astype(float)
            testing_stack.append(listrow)
    testing_matrix = np.vstack(testing_stack)

    # predict outcomes of testing data using training data

    # make sum function
    def sumfun(x):
        return sum(x)

    trainlen = len(training_matrix)
    for line in testing_matrix:
        eval_stack = []
        for i in range(trainlen):
            eval_stack.append(line)
        eval_matrix = np.vstack(eval_stack)
        euclid_matrix = (eval_matrix - training_matrix) ** 2
        euclids = np.apply_along_axis(sumfun, axis=1, arr=euclid_matrix)
        distances_dict = dict(zip(euclids, training_goal))
        distances_ordered = collections.OrderedDict(sorted(distances_dict.items()))
        top_values = list(distances_ordered.values())
        ktop = top_values[0:k]
        #print(ktop)
        y = 0
        n = 0
        for i in ktop:
            if i =="yes\n" or i =="yes":
                y+=1
            else:
                n+=1
        if y>=n:
            predicted_goal.append("yes")
        else:
            predicted_goal.append("no")

    return predicted_goal