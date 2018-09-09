import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1 # margin of the SVM
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  dim, num_train = X.shape
  # print X.dtype
  num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
  # print "Num Classes: ", num_classes
  if W is None:
    # lazily initialize W
    W = np.random.randn(num_classes, dim) * 0.001

  loss_image_arr = np.empty([num_train, num_classes])

  # Information about various dimensions
  # print "Num Dimensions: ", dim, "Num Samples: ", num_train, "Num Classes: ", num_classes

  score_matrix = np.matmul(W, X)
  for i in range(num_train):

    # Sanity Check for the sizes of he matrices after multiplication
    # All rows in score_matrix represents the score of an image in a class
    # print "Weight Matrix Shape: ", self.W.shape, "Score Matrix Shape: ", score_matrix.shape

    for j in range(num_classes):
      if (j!=y[i]):
        loss_image_arr[i, j] = (max(0, score_matrix[j, i] - score_matrix[y[i], i] + delta))

  reg_loss  = reg * np.sum(np.square(W))
  loss = np.sum(loss_image_arr)/num_train + reg_loss

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  binary_matrix = loss_image_arr>0
  # print "Binary Matrix: ", binary_matrix.shape, "X_batch: ", X_batch.shape

  dW = np.transpose(np.matmul(X, binary_matrix))
  dW = dW/num_train
  # print "Iteration -- ", "Loss: ", loss_iter , "Gradient Shape: ", dW.shape, "Weight Shape: ", self.W.shape

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
