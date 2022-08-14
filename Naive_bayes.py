import collections
import statistics
from statistics import mode
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
import math

def pdf(x,mean,sd):
  res = ((1/(sd* math.sqrt(2*np.pi)))*np.exp(-((x-mean)**2/(2*(sd**2)))))
  return res

def classify_nb(training_filename, testing_filename):
    # convert data into numpy vectors

    # convert training data
    y_training_stack = []
    n_training_stack = []
    training_goal = []
    testing_stack = []
    predicted_goal = []
    with open(training_filename) as training_data:
        for line in training_data.readlines():
            linerow = line.split(",")
            ynval = linerow[-1]
            training_goal.append(linerow[-1])
            linerow.pop()
            listrow = np.array(linerow).astype(float)
            if ynval == "yes\n" or ynval == "yes":
                y_training_stack.append(listrow)
            else:
                n_training_stack.append(listrow)
    y_training_matrix = np.vstack(y_training_stack)
    n_training_matrix = np.vstack(n_training_stack)

    # convert testing data
    with open(testing_filename) as testing_data:
        for line in testing_data.readlines():
            linerow = line.split(",")
            #if "\n" in linerow[-1]:
            #    linerow.pop()
            listrow = np.array(linerow).astype(float)
            testing_stack.append(listrow)
    testing_matrix = np.vstack(testing_stack)
    y_row_tm, y_col_tm = y_training_matrix.shape
    n_row_tm, n_col_tm = n_training_matrix.shape
    y_mu_sd = []
    n_mu_sd = []
    ych = (y_row_tm/(y_row_tm+n_row_tm))
    nch = (n_row_tm/(y_row_tm+n_row_tm))
    for i in range(y_col_tm):
        col_vals = list(y_training_matrix[:,i])
        mean_col = sum(col_vals)/len(col_vals)
        st_col = 0
        for i in col_vals:
          st_col+=(i-mean_col)**2
        st_col = math.sqrt(st_col/(len(col_vals)-1))
        y_mu_sd.append([mean_col,st_col])

    for i in range(n_col_tm):
        col_vals = list(n_training_matrix[:,i])
        mean_col = sum(col_vals)/len(col_vals)
        st_col = 0
        for i in col_vals:
          st_col+=(i-mean_col)**2
        st_col = math.sqrt(st_col/(len(col_vals)-1))
        n_mu_sd.append([mean_col,st_col])
    for line in testing_matrix:
        y_log_likelyhoods = []
        n_log_likelyhoods = []
        for i in range(len(line)):
            nm = 2
            n_log_likelyhoods.append(np.log(pdf(line[i],n_mu_sd[i][0],n_mu_sd[i][1])))
            y_log_likelyhoods.append(np.log(pdf(line[i],y_mu_sd[i][0],y_mu_sd[i][1])))
        y_log_likelyhoods.append(np.log(len(y_training_stack)/(len(n_training_stack)+len(y_training_stack))))
        n_log_likelyhoods.append(np.log(len(n_training_stack)/(len(n_training_stack)+len(y_training_stack))))
        if sum(y_log_likelyhoods) >= sum(n_log_likelyhoods):
            predicted_goal.append("yes")
        else:
            predicted_goal.append("no")
    # find mean and standard deviation for each column


    # predict outcomes of testing data using training data

    # make sum function
    return predicted_goal
#print(classify_nb("mainfile.txt","testingfile.txt"))