import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  
  for i in range(num_train):
    s = X[i,:].dot(W)
    softmax = np.exp(s) / np.sum(np.exp(s))
  
    loss += -np.log(softmax[y[i]])
  
    for j in range(W.shape[1]):
        dW[:,j] += X[i] * softmax[j]
    dW[:, y[i]] -= X[i]
        
  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dW += 2 * W * reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  s = X.dot(W)
  s = s - s.max()
  s = np.exp(s)
  
  sum_of_s = np.sum(s, axis = 1)
  softmax = s[range(num_train), y]
  loss = softmax / sum_of_s
  loss = -np.sum(np.log(loss)) / num_train + reg * np.sum(W * W)
  
  x = np.divide(s, sum_of_s.reshape(num_train, 1))
  x[range(num_train), y] = -(sum_of_s - softmax) / sum_of_s
  
  dW = X.T.dot(x)
  dW /= num_train
  dW += 2 * W * reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

