import numpy as np
import theano
import theano.tensor as T
import random
from itertools import izip

THRESHOLD = 0
LEARNING_RATE = np.float32(0.001)
TOTAL_LAYERS = 3 
BATCH_SIZE = 128
OUTPUT_WIDTH = 1
INPUT_WIDTH = 2
LAYER_WIDTH = 100
MOMENTUM = 0.9

def sigmoid(z):
    a = 1 / (1 + T.exp(-z))
    return a

def softmax(z):
    y = T.exp(z) / T.sum(T.exp(z)) #transpose output
    return y

class RNN:

    def __init__(self , layers = 1):
        x_seq = T.matrix()
        y_hat = T.scalar()

        Wi = theano.shared(np.random.uniform(-0.1, 0.1, size = (INPUT_WIDTH, LAYER_WIDTH)))
        bh = theano.shared(np.random.uniform(-0.1, 0.1, size = (LAYER_WIDTH))) 
        Wh = theano.shared(np.random.uniform(-0.1, 0.1, size = (LAYER_WIDTH, LAYER_WIDTH)))
        Wo = theano.shared(np.random.uniform(-0.1, 0.1, size = (LAYER_WIDTH, OUTPUT_WIDTH)))
        bo = theano.shared(np.random.uniform(-0.1, 0.1, size = (OUTPUT_WIDTH)))
        parameters = [Wi, bh, Wh, Wo, bo]
            
        def step(x_t, a_tm1, y_tm1): 
            a_t = sigmoid(T.dot(x_t, Wi) + T.dot(a_tm1, Wh) + bh)
            y_t = softmax(T.dot(a_t, Wo)) + bo 
            #y_t = T.dot(a_t, Wo) + bo
            return a_t, y_t

        a_0 = theano.shared(np.ones(LAYER_WIDTH))
        y_0 = theano.shared(np.ones(OUTPUT_WIDTH))
        
        [a_seq, y_seq], _ = theano.scan(
                            step, 
                            sequences = x_seq, 
                            outputs_info = [a_0, y_0], 
                            truncate_gradient=-1
                            )
        

        y_seq_last = y_seq[-1][0]
        cost = T.sum( ( y_seq_last - y_hat )**2 ) 
        gradients = T.grad(cost, parameters)

        def myUpdate(parameters, gradients):
            parameter_updates =  [(p, p - LEARNING_RATE * g) for p, g in izip(parameters , gradients)] 
            return parameter_updates
        
        self.train = theano.function(
            inputs  = [x_seq, y_hat],
            allow_input_downcast= True,
            updates = myUpdate(parameters, gradients),
            outputs = cost) 

        # Validating function
        self.valid = theano.function(
            inputs  = [x_seq],
            allow_input_downcast= True,
            outputs = y_seq_last)
        

