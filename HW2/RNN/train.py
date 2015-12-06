import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time
from rnn import RNN

rnn = RNN() 


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# input
N_INPUT = 2
# output
N_OUTPUT = 1


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    '''
    Generate a sequences for the "add" task, e.g. the target for the
    following
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``
    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample (a scalar).
    '''
    # Generate x_seq
    length = np.random.randint(min_length, max_length)  
    x_seq = np.concatenate([np.random.uniform(size=(length, 1)),
                        np.zeros((length, 1))],
                       axis=-1)
    # Set the second dimension to 1 at the indices to add
    x_seq[np.random.randint(length/10), 1] = 1
    x_seq[np.random.randint(length/2, length), 1] = 1
    # Multiply and sum the dimensions of x_seq to get the target value
    y_hat = np.sum(x_seq[:, 0]*x_seq[:, 1])
    return x_seq, y_hat


for i in range(1000):
    x_seq, y_hat = gen_data()
    print "y_hat: ", y_hat , " y: " , rnn.valid(x_seq)
    print "iteration:", i, "cost:",  rnn.train(x_seq,y_hat)

