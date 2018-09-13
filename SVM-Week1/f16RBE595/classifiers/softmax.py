import numpy as np
from random import shuffle

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

  dim, num_train = X.shape
  # print X.dtype
  num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
  # print "Num Classes: ", num_classes
  if W is None:
    # lazily initialize W
    W = np.random.randn(num_classes, dim) * 0.0001

  loss_image_arr = np.empty(num_train)

  # Information about various dimensions
  # print "Num Dimensions: ", dim, "Num Samples: ", num_train, "Num Classes: ", num_classes

  score_matrix = np.matmul(W, X)
  for i in range(num_train):
    sample_sum = 0.0
    for j in range(num_classes):
      sample_sum += score_matrix[j, i]

    loss_image_arr[i] = -score_matrix[y[i], i] + np.log(sample_sum)

  reg_loss  = reg * np.sum(np.square(W))
  loss = np.sum(loss_image_arr)/num_train + reg_loss

  new_loss_mat = np.matmul(np.transpose(loss_image_arr), loss_image_arr)
  binary_matrix = new_loss_mat>0
  # print "Binary Matrix: ", binary_matrix.shape, "X_batch: ", X_batch.shape

  dW = -np.transpose(np.matmul(np.matrix(X), binary_matrix))
  dW = dW/num_train
  # print "Iteration -- ", "Loss: ", loss_iter , "Gradient Shape: ", dW.shape, "Weight Shape: ", self.W.shape

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
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

