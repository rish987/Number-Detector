"""
Author:     Rish Vaishnav
Date:       1/5/2017
File:       detect_num_log.py

An implementation of one-vs-all logistic regression to classify handwritten
digits.
"""
import time
from numpy import *
from scipy.optimize import fmin_bfgs

def sigmoid ( data ):
    """
    Returns a matrix containg the sigmoid of each element in a given matrix of
    data.

    data ( matrix of real numbers ) - the data to take the sigmoid of
    """
    # return the sigmoid of the data
    return 1 / ( 1 + exp( -data ) )

def get_reg_log_cost ( params_in, train_set_inps_in, train_set_outps_in, \
    reg_param ):
    """
    Returns the regularized logistic cost of a model, given the model's
    parameters and the training set inputs and outputs.

    params_in (row vector of real numbers) - parameters of model
    train_set_inps_in ( matrix of real numbers ) - training set inputs, without
        intercept ( bias ) term
    train_set_outps_in ( vector of real numbers ) - training set outputs
    reg_param ( positive real number ) - regularization parameter
    """
    # convert params to column vector
    params = matrix( params_in ).T

    # convert training set to matrices
    train_set_inps = matrix( train_set_inps_in )
    train_set_outps = matrix( train_set_outps_in )

    # number of training examples
    m = float( len( train_set_inps ) )

    # training set inputs with bias term
    train_set_inps_adj = insert( train_set_inps, 0, 1, axis = 1 )

    # get the predicted outputs
    pred_outps = sigmoid( train_set_inps_adj * params )

    # list of individual costs for each training example
    costs = multiply( -train_set_outps, log( pred_outps ) ) - \
        multiply( ( 1 - train_set_outps ), log( 1 + 1e-15 - pred_outps ) )

    # cost, not including regularization
    cost = ( 1 / m ) * sum( costs )
    # cost, including regularization
    cost += ( reg_param / ( 2 * m ) ) * \
           sum( multiply( params[ 1: ], params[ 1: ] ) )

    # return the cost
    return cost

def get_reg_log_grad ( params_in, train_set_inps_in, train_set_outps_in, \
    reg_param  ):
    """
    Returns the regularized logistic gradient for a given set of parameters
    and training data.

    params_in (row vector of real numbers) - parameters of model
    train_set_inps_in ( matrix of real numbers ) - training set inputs, without
        intercept ( bias ) term
    train_set_outps_in ( vector of real numbers ) - training set outputs
    reg_param ( positive real number ) - regularization parameter
    """
    # convert params to column vector
    params = matrix( params_in ).T

    # convert to matrices
    train_set_inps = matrix( train_set_inps_in )
    train_set_outps = matrix( train_set_outps_in )

    # size of training data
    m = float( len( train_set_inps ) )

    # training set inputs with bias terms
    adj_train_set_inps = insert( train_set_inps, 0, 1, axis = 1 )
    
    #( adj_train_set_inps.T * diffs ) / m get the predicted outputs
    pred_outps = sigmoid( adj_train_set_inps * params )

    # the differences
    diffs = pred_outps - train_set_outps

    # the gradients, without regularization
    grads = ( adj_train_set_inps.T * diffs ) / m
    # the gradients, with regularization
    grads[ 1: ] = grads[ 1: ] + ( reg_param / m ) * params[ 1: ]

    # return the flattened gradients
    return array( grads ).flatten()

# read in data
data = matrix( loadtxt( "num_data.txt", delimiter="," ) )

# sample x data
data_x = data[ :, :-1 ]

# sample y data
data_y = data[ :, -1 ]

# to hold the parameters for each number
params = []

# regularization parameter
reg = 1

# go through all of the numbers
for num in range( 10 ):

    #TODO
    print "\nNumber: " + str( num ) + "\n"
    
    # this y data
    this_y = equal( data_y, num ).astype( int )

    # perform optimization
    params.append( fmin_bfgs( lambda x: get_reg_log_cost( x, data_x, \
        this_y, reg ), zeros(401), fprime = \
        lambda x: get_reg_log_grad( x, data_x, this_y, reg ) ) )

# matrix of parameters
params_mx = matrix( params )

# read in test data
test_data = matrix( loadtxt( "test_data.txt", delimiter="," ) )
#test_data = data_x

# test data inputs with bias term
test_data_adj = insert( test_data, 0, 1, axis = 1 )

# get probabilites from each hypothesis for each training example
test_probs = sigmoid( test_data_adj * params_mx.T )

# get predictions
predictions = argmax( test_probs, axis = 1 )

#TODO
print predictions
